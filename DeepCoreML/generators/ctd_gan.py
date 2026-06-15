import numpy as np
import math

import torch
import torch.nn as nn

from tqdm import tqdm

from DeepCoreML import paths
from DeepCoreML.TabularTransformer import TabularTransformer
from DeepCoreML.generators.gan_discriminators import Critic
from DeepCoreML.generators.gan_generators import ctGenerator
from DeepCoreML.generators.GAN_Synthesizer import GANSynthesizer
from DeepCoreML.generators.ctd_clusterer import ctdClusterer
from DeepCoreML.generators.ctd_classifier import ctdClassifier, train_classifier
from DeepCoreML.generators.ctd_datasampler import ctdDataSampler

# import DeepCoreML.paths as paths
torch.set_printoptions(threshold=20000)


class ctdGAN(GANSynthesizer):
    """
    ctdGAN implementation

    ctdGAN conditionally generates tabular data with the aim of confronting class imbalance. The model uses both
    cluster and class labels for training. It applies a cluster-aware data transformation mechanism and introduces
    a loss function that penalizes the generation of samples with incorrect cluster and class labels. New data
    instances are generated via a probabilistic sampling strategy.
    """

    def __init__(self, discriminator=(128, 128), generator=(256, 256), embedding_dim=128, epochs=300, batch_size=32,
                 pac=1, lr=2e-4, decay=1e-6, sampling_strategy='auto', use_classifier=True,
                 scaler='mms11', cluster_method='kmeans', max_clusters=20, alpha_k=0.02, random_state=0):
        """
        ctdGAN initializer

        Args:
            discriminator (tuple): a tuple with number of neurons for each fully connected layer of the model's Critic.
                The tuple elements determine the dimensionality of the output of each layer.
            generator (tuple): a tuple with number of neurons for each fully connected layer of the model's Generator.
                The tuple elements determine the dimensionality of the output of each residual block of the Generator.
            embedding_dim (int): Size of the normally distributed latent vector passed to the Generator.
            epochs (int): The number of training epochs.
            batch_size (int): The number of data instances per training batch. Must be α multiple of `pac`.
            pac (int): The number of samples to group together as input to the Critic.
            lr (real): The value of the learning rate parameter for the Generator/Critic Adam optimizers.
            decay (real): The value of the weight decay parameter for the Generator/Critic Adam optimizers.
            sampling_strategy (string or dictionary): How the model generates samples:

                * 'auto': balance the dataset by oversampling the minority classes.
                * 'balance-clusters': balance the dataset by balancing its clusters.
                * 'create-new': create a new dataset with the same class distribution as the one that was trained with.
                * dict: a dictionary that indicates the number of samples to be generated from each class.
            use_classifier: Train a classifier to check whether the generated samples are from realistic classes.
                Penalize the Generator when it produces samples from incorrect classes.
            scaler (string): A descriptor that defines a transformation on the cluster's data. Values:

               * '`None`'  : No transformation takes place; the data is considered immutable
               * '`stds`'  : Standard scaler
               * '`mms01`' : Min-Max scaler in the range (0,1)
               * '`mms11`' : Min-Max scaler in the range (-1,1) - so that data is suitable for tanh activations
               * '`yeo`':  Yeo-Johnson Power Transformer
            max_clusters (int): The maximum number of clusters to create.
            alpha_k: Sensitivity parameter for the number of clusters estimator.
            random_state (int): Seed the random number generators. Use the same value for reproducible results.
        """
        super().__init__("ctdGAN", embedding_dim, discriminator, generator, pac, epochs, batch_size,
                         lr, lr, decay, decay, sampling_strategy, random_state)


        if scaler not in ('None', 'none', 'stds', 'mms01', 'mms11', 'yeo', 'glob-mms11', 'glob-stds', 'glob-vgm'):
            self._scaler = 'mms11'
        else:
            self._scaler = scaler

        # clustered_transformer performs clustering and data transformation.
        self._clustered_transformer = None

        # discrete_transformer performs dataset-wise one-hot-encoding of the categorical columns
        self._discrete_transformer = None

        self._data_sampler = None

        self._cluster_method = cluster_method
        self._max_clusters = max_clusters
        self._alpha_k = alpha_k

        self._use_classifiers = use_classifier
        self._n_clusters = 0
        self._n_classes = 0
        self._categorical_columns = []

        self.class_col_start_index = 0
        self.class_col_end_index = 0
        self.cluster_col_start_index = 0
        self.cluster_col_end_index = 0

    @staticmethod
    def _gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits (array(…, num_features)): Un-normalized log probabilities
            tau: Non-negative scalar temperature/
            hard (bool): If True, the returned samples will be transformed to one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd.
            dim (int): A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._discrete_transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def cluster_transform(self, x_train, y_train, categorical_columns):
        """
        Perform clustering and transform the data in the generated clusters.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
            categorical_columns: The discrete columns in the dataset. The last 2 columns indicate the cluster and class.

        Returns:
            A tensor with the preprocessed data.
        """
        self._categorical_columns = list(categorical_columns)
        self._n_classes = len(set(y_train))
        self._input_dim = x_train.shape[1]
        continuous_columns = [c for c in range(self._input_dim) if c not in self._categorical_columns]

        # ====== Initialize and fit the Clustered Transformer object that: i) partitions the real space, and
        # ====== ii) performs data transformations (scaling, PCA, outlier detection, etc.)
        self._samples_per_class = np.unique(y_train, return_counts=True)[1]

        self._clustered_transformer = ctdClusterer(cluster_method=self._cluster_method, max_clusters=self._max_clusters,
                                                   alpha_k=self._alpha_k, scaler=self._scaler,
                                                   samples_per_class=self._samples_per_class,
                                                   continuous_columns=tuple(continuous_columns),
                                                   categorical_columns=tuple(self._categorical_columns),
                                                   embedding_dim=self.embedding_dim_, random_state=self._random_state)

        train_data = self._clustered_transformer.perform_clustering(x_train, y_train, self._n_classes, self.pac_)

        train_classes = train_data[:, -1]

        # print(train_data[0:100, :])
        self._n_clusters = self._clustered_transformer.num_clusters_

        # ====== Append the cluster and class labels to the collection of discrete columns
        self._categorical_columns.append(self._input_dim)
        self._categorical_columns.append(self._input_dim + 1)

        # ====== Transform the discrete columns only; the continuous columns have been scaled at cluster-level.
        self._discrete_transformer = TabularTransformer(cont_normalizer=self._scaler, clip=True)
        self._discrete_transformer.fit(train_data, self._categorical_columns)
        ret_data = self._discrete_transformer.transform(train_data)

        self._data_sampler = ctdDataSampler(ret_data, self._discrete_transformer.output_info_list, True)

        # Return the data for ctdGAN training
        return ret_data, train_classes

    def cond_loss(self, generated_data, generated_data_after_act, c, m, lamda_cls):
        """Compute the cross entropy loss on the fixed discrete column."""
        num_generated_samples = generated_data.size()[0]

        discrete_loss = []
        #lamda_cls = 1.0

        st = 0
        st_c = 0
        for column_info in self._discrete_transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    # print("Start:", st, ", End: ", ed, ", Generated Data:", generated_data[:, st:ed])
                    # print("Start_Col:", st_c, ", End_Col: ", ed_c, ", CondVec:", c[:, st_c:ed_c])
                    gen = generated_data[:, st:ed]
                    gen_c = torch.argmax(gen, dim=1)

                    lat = torch.argmax(c[:, st_c:ed_c], dim=1)

                    # Penalize the incorrect cluster generation
                    if st == self.cluster_col_start_index and ed == self.cluster_col_end_index:

                        if self._use_classifiers:
                            predicted_clusters = self.CU_(generated_data_after_act[:, : self.cluster_col_start_index])
                            classifier_loss = nn.CrossEntropyLoss()(predicted_clusters, lat)
                            tmp = nn.functional.cross_entropy(gen, lat, reduction='none') + lamda_cls * classifier_loss
                        else:
                            mis_clustered = np.sum([1 for s in range(num_generated_samples) if gen_c[s] != lat[s]])
                            #print("Lat Clusters:\n", lat, "\nGen Clusters:\n", gen, "(", gen_c, ")")
                            #print("\tMisclustered samples: ", mis_clustered)
                            #beta = 1.0
                            beta = 1.0 + mis_clustered / num_generated_samples
                            tmp = beta * nn.functional.cross_entropy(gen, lat, reduction='none')

                    # Penalize the incorrect class generation
                    elif st == self.class_col_start_index and ed == self.class_col_end_index:
                        #print("latent classes:", lat)
                        #print("generated classes:", gen_c)

                        if self._use_classifiers:
                            predicted_classes = self.C_(generated_data_after_act[:, :self.cluster_col_start_index]) # cls4

                            classifier_loss = nn.CrossEntropyLoss()(predicted_classes, lat)
                            tmp = nn.functional.cross_entropy(gen, lat, reduction='none') + lamda_cls * classifier_loss
                        else:
                            tmp = nn.functional.cross_entropy(gen, lat, reduction='none')
                    else:
                        tmp = nn.functional.cross_entropy(gen, lat, reduction='none')

                    # tmp = nn.functional.cross_entropy(gen, lat, reduction='none')
                    # print("Temp=", tmp)
                    # a = input('').split(" ")[0]
                    # print("Original: ")
                    discrete_loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(discrete_loss, dim=1)  # noqa: PD013
        ret_loss = (loss * m).sum() / generated_data.size()[0]

        return ret_loss

    def _train(self, x_train, y_train, categorical_columns=(), store_losses=None):
        """
        ctdGAN training process. The Generator and the Critic are trained jointly in the traditional adversarial
        fashion by optimizing `loss_function`.

        Args:
            x_train (NumPy array): The training data instances.
            y_train (NumPy array): The classes of the training data instances.
            categorical_columns: The columns to be considered as categorical
            store_losses: The file path where the values of the Discriminator and Generator loss functions are stored.
        """

        # Step 1: Preprocessing
        # Modify the size of the batch to align with self.pac_
        factor = self._batch_size // self.pac_
        batch_size = factor * self.pac_

        # Step 2: Perform clustering and transform the data in the generated clusters.
        training_data, training_classes = self.cluster_transform(x_train, y_train,
                                                                 categorical_columns=categorical_columns)

        self.class_col_start_index = self._discrete_transformer.output_dimensions - self._n_classes
        self.class_col_end_index = self.class_col_start_index + self._n_classes
        self.cluster_col_start_index = self._discrete_transformer.output_dimensions - self._n_clusters - self._n_classes
        self.cluster_col_end_index = self.cluster_col_start_index + self._n_clusters

        real_space_dimensions = self._discrete_transformer.output_dimensions + self._discrete_transformer.ohe_dimensions
        latent_space_dimensions = self.embedding_dim_ + self._discrete_transformer.ohe_dimensions

        # print("Class Col Start:", self.class_col_start_index, ", Class Col End:", self.class_col_end_index)
        # print("Cluster Col Start", self.cluster_col_start_index, ", Cluster Col End:", self.cluster_col_end_index)

        # Data Sampler
        self._data_sampler = ctdDataSampler(training_data, self._discrete_transformer.output_info_list, True)

        # Step 3: Set up the model networks: Discriminator, Generator and Classifier
        # 3A: Discriminator & Optimizer
        self.D_ = Critic(input_dim=real_space_dimensions, discriminator_dim=self.D_Arch_, pac=self.pac_).to(
            self._device)
        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(), lr=self._disc_lr, weight_decay=self._disc_decay,
                                             betas=(0.5, 0.9))

        # 3B: Generator & Optimizer
        self.G_ = ctGenerator(embedding_dim=latent_space_dimensions, architecture=self.G_Arch_,
                              data_dim=real_space_dimensions).to(self._device)
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(), lr=self._gen_lr, weight_decay=self._gen_decay,
                                             betas=(0.5, 0.9))

        if self._use_classifiers:
            # --------------------------
            # Classifier for classes
            num_rows = training_data.shape[0]
            num_train_rows = math.ceil(0.8 * num_rows)

            # CLS4
            x_cl_train = (torch.Tensor(training_data[ 0 : num_train_rows, 0 : self.cluster_col_start_index])
                .to(dtype=torch.float32).to(device=self._device))
            x_cl_val = (torch.Tensor(training_data[ num_train_rows : num_rows, 0 : self.cluster_col_start_index])
                .to(dtype=torch.float32).to(device=self._device))

            y_cl_train = (torch.Tensor(np.argmax(training_data[0 : num_train_rows, self.class_col_start_index :], axis=1))
                .long().to(device=self._device))
            y_cl_val = (torch.Tensor(np.argmax(training_data[num_train_rows : num_rows, self.class_col_start_index :], axis=1))
                .long().to(device=self._device))

            # print("\nNum Rows:\n", num_rows, "\n", x_cl_train.shape, "-", x_cl_val.shape, "\n", y_cl_train.shape, "-", y_cl_val.shape)

            self.C_ = train_classifier(x_train=x_cl_train, y_train=y_cl_train, x_val=x_cl_val, y_val=y_cl_val,
                                       hidden_dims=(128, 256, 256, 128), input_dim=self.cluster_col_start_index,
                                       num_classes=self._n_classes, batch_size=64, epochs=30, lr=1e-3)

            # Freeze the classifier gradients
            for p in self.C_.parameters():
                p.requires_grad = False
            self.C_.eval()

            # --------------------------
            # Classifier for clusters
            y_clu_train = (torch.Tensor(np.argmax(training_data[0 : num_train_rows, self.cluster_col_start_index : self.class_col_start_index ], axis=1))
                           .long().to(device=self._device))
            y_clu_val = (torch.Tensor(np.argmax(training_data[num_train_rows : num_rows, self.cluster_col_start_index : self.class_col_start_index ], axis=1))
                         .long().to(device=self._device))

            self.CU_ = train_classifier(x_train=x_cl_train, y_train=y_clu_train, x_val=x_cl_val, y_val=y_clu_val,
                                        hidden_dims=(128, 256, 256, 128), input_dim=self.cluster_col_start_index,
                                        num_classes=self._n_clusters, batch_size=64, epochs=30, lr=1e-3)

            # Freeze the classifier gradients
            for pU in self.CU_.parameters():
                pU.requires_grad = False

            self.CU_.eval()

        # Step 4: Start ctdGAN training loop
        losses = []
        it = 0
        steps_per_epoch = max(len(training_data) // self._batch_size, 1)
        mean = torch.zeros(self._batch_size, self.embedding_dim_, device=self._device)
        std = mean + 1
        lamda_cls = 0.01
        for epoch in tqdm(range(self._epochs), desc="ctdGAN Training     "):
            if epoch > 20:
                lamda_cls = min(1.0, lamda_cls * 1.02)
            else:
                lamda_cls = 0.01

            for id_ in range(steps_per_epoch):
                it += 1
                loss_d, loss_g = self.train_batch(training_data, mean, std, lamda_cls, batch_type='normal')
                if store_losses is not None:
                    losses.append((it, epoch + 1, loss_d.detach().cpu().item(), loss_g.detach().cpu().item()))

            # loss_d, loss_g = self.train_batch(training_data, mean, std, lamda_cls, batch_type='class')
            #if store_losses is not None:
            #    losses.append((it, epoch + 1, loss_d.detach().cpu().item(), loss_g.detach().cpu().item()))

        if store_losses is not None:
            self.plot_losses(losses, store_losses)


    def train_batch(self, training_data, mean, std, lamda_cls, batch_type='normal'):
        fakez = torch.normal(mean=mean, std=std)

        condvec = self._data_sampler.sample_condvec(self._batch_size, batch_type, self._n_classes)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
            real = self._data_sampler.sample_data(training_data, self._batch_size, col, opt, batch_type)
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self._device)
            m1 = torch.from_numpy(m1).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)

            perm = np.arange(self._batch_size)
            np.random.shuffle(perm)
            real = self._data_sampler.sample_data(training_data, self._batch_size, col[perm], opt[perm], batch_type)
            c2 = c1[perm]

        fake = self.G_(fakez)
        fakeact = self._apply_activate(fake)

        real = torch.from_numpy(real.astype('float32')).to(self._device)

        if c1 is not None:
            fake_cat = torch.cat([fakeact, c1], dim=1)
            real_cat = torch.cat([real, c2], dim=1)
        else:
            real_cat = real
            fake_cat = fakeact

        y_fake = self.D_(fake_cat)
        y_real = self.D_(real_cat)

        pen = self.D_.calc_gradient_penalty(real_cat, fake_cat, self._device, self.pac_)
        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

        self.D_optimizer_.zero_grad(set_to_none=False)
        pen.backward(retain_graph=True)
        loss_d.backward()
        self.D_optimizer_.step()

        # Generator training
        # Sample from normal distribution
        fakez = torch.normal(mean=mean, std=std)

        # Create a new conditional vector only for normal batches. For the batches of type 'class' use the same
        # conditional vector as the one that we used for the discriminator. These batches strengthen the ability
        # of a Generator to create samples from a particular class.
        if batch_type == 'normal':
            condvec = self._data_sampler.sample_condvec(self._batch_size, 'normal', -1)

        if condvec is None:
            c1, m1, col, opt = None, None, None, None
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self._device)
            m1 = torch.from_numpy(m1).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)

        fake = self.G_(fakez)
        fakeact = self._apply_activate(fake)

        if c1 is not None:
            y_fake = self.D_(torch.cat([fakeact, c1], dim=1))
        else:
            y_fake = self.D_(fakeact)

        if condvec is None:
            cross_entropy = 0
        else:
            cross_entropy = self.cond_loss(fake, fakeact, c1, m1, lamda_cls)

        loss_g = -torch.mean(y_fake) + cross_entropy

        self.G_optimizer_.zero_grad(set_to_none=False)
        loss_g.backward()
        self.G_optimizer_.step()

        return loss_d, loss_g


    def fit(self, x_train, y_train):
        """Invokes the GAN training process.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
        """
        self._train(x_train, y_train)

    def sample(self, num_samples, condition_column=None, condition_value=None, sec_condition_column=None, sec_condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            num_samples (int): Number of rows to sample.
            condition_column: Name of a discrete column.
            condition_value: Name of the category in the condition_column which we wish to increase the
                probability of happening.
            sec_condition_column: Name of a secondary discrete column.
            sec_condition_value: Name of the category in the sec_condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        num_generated_samples, num_rejected_samples, num_retries, max_retries = 0, 0, 0, 500
        reconstructed_samples = []

        if condition_column is not None and condition_value is not None:
            condition_info = self._discrete_transformer.convert_column_name_value_to_id(condition_column, condition_value)
            if sec_condition_column is not None and sec_condition_value is not None:
                condition_info_2 = self._discrete_transformer.convert_column_name_value_to_id(sec_condition_column, sec_condition_value)
            else:
                condition_info_2 = None

            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, condition_info_2, self._batch_size)
        else:
            global_condition_vec = None

        steps = num_samples // self._batch_size + 1

        # Keep generating samples, until we reach the requested number of num_samples
        while num_generated_samples < num_samples:
            num_retries += 1
            generated_data = []

            for i in range(steps):
                mean = torch.zeros(self._batch_size, self.embedding_dim_)
                std = mean + 1
                fakez = torch.normal(mean=mean, std=std).to(self._device)

                if global_condition_vec is not None:
                    condvec = global_condition_vec.copy()
                else:
                    condvec = self._data_sampler.sample_original_condvec(self._batch_size)

                if condvec is None:
                    pass
                else:
                    c1 = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.G_(fakez)
                fakeact = self._apply_activate(fake)

                generated_data.append(fakeact.detach().cpu().numpy())

            generated_data = np.concatenate(generated_data, axis=0)
            generated_samples = self._discrete_transformer.inverse_transform(generated_data)
            for s in range(self._batch_size):
                z = generated_samples[s].reshape(1, -1)
                generated_class = int(z[0, z.shape[1] - 1])
                generated_cluster = int(z[0, z.shape[1] - 2])

                #print("Generated class:", generated_class)
                #print("Generated cluster:", generated_cluster, '-', self._clustered_transformer.get_cluster(generated_cluster).get_label())
                # In case we do not care about the generated cluster (Sample uniformly from all clusters)
                if sec_condition_value is None:
                    sec_condition_value = generated_cluster

                if generated_class == condition_value and generated_cluster == sec_condition_value:
                    num_generated_samples += 1
                    if num_generated_samples > num_samples:
                        return_samples = np.vstack(reconstructed_samples)
                        #print("\t\tPerfectly created", return_samples.shape[0], "samples from class:", condition_value,
                        #      "in cluster:", sec_condition_value, "- rejected:", num_rejected_samples, "- retries:", num_retries)
                        return return_samples

                    reconstructed_sample = self._clustered_transformer.get_cluster(generated_cluster).inverse_transform(z)
                    reconstructed_samples.append(reconstructed_sample)
                    #print("Sample", s, "- Gen:", z, " ===>", reconstructed_sample)
                else:
                    num_rejected_samples += 1

            # If the maximum number of attempts has been exhausted, then exit the loop.
            # We will be generating fewer than the requested samples.
            if num_retries > max_retries:
                break

        return_samples = np.vstack(reconstructed_samples)
        '''
        if return_samples.shape[0] < num_samples:
            print("\t\tIncompletely created ", return_samples.shape[0], "samples (req: ", num_samples , ") from class:",
                  condition_value, "in cluster:", sec_condition_value, "- rejected:", num_rejected_samples,
                  "- retries:", num_retries)
        else:
            print("\t\tPerfectly created", return_samples.shape[0], "samples from class:", condition_value,
                  "in cluster:", sec_condition_value, "- rejected:", num_rejected_samples, "- retries:", num_retries)
        '''
        return return_samples


    def fit_resample(self, x_train, y_train, categorical_columns=()):
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. The function renders ctdGAN
        compatible with the `imblearn`'s interface, allowing its usage in over-sampling/under-sampling pipelines.

        In the `fit` part, the input dataset is used for training.
        In the `resample` part, the model generates synthetic data according to the value of `self._sampling_strategy`:

        - 'auto': balance the dataset by oversampling the minority classes.
        - 'balance-clusters': balance the dataset by balancing its clusters.
        - 'create-new': create a new dataset with the same class distribution as the one that was trained with
        - dict: a dictionary that indicates the number of samples to be generated from each class

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
            categorical_columns: The columns to be considered as categorical

        Returns:
            x_resampled: The training data instances + the generated data instances.
            y_resampled: The classes of the training data instances + the classes of the generated data instances.
        """

        # Train ctdGAN with the input data
        #self._train(x_train, y_train, categorical_columns=categorical_columns, store_losses=paths.output_path_loss)
        self._train(x_train, y_train, categorical_columns=categorical_columns, store_losses=None)

        x_resampled = np.copy(x_train)
        y_resampled = np.copy(y_train)

        # Class column: The last column + 1. This +1 derives from the insertion of the cluster label column
        cluster_column = x_train.shape[1]
        class_column = x_train.shape[1] + 1

        # auto mode: Use ctdGAN to equalize the number of samples per class. This is achieved by generating samples
        # of the minority classes (i.e. we perform oversampling).
        if self._sampling_strategy == 'auto':
            majority_class = np.array(self._samples_per_class).argmax()
            num_majority_samples = np.max(np.array(self._samples_per_class))

            # Perform oversampling
            for cls in tqdm(range(self._n_classes), desc="ctdGAN Sampling     "):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._samples_per_class[cls]

                    p_matrix = self._clustered_transformer.probability_matrix_[cls]
                    samples_to_generate_per_cluster = np.rint(p_matrix * samples_to_generate).astype(np.int32)
                    # print("Total samples:", samples_to_generate)
                    # print("Class:", cls, "- Total samples to create (per cluster):", samples_to_generate_per_cluster)

                    for cluster in range(self._n_clusters):
                        if samples_to_generate_per_cluster[cluster] > 1:
                            # Generate the appropriate number of samples to equalize cls with the majority class.
                            generated_samples = self.sample(num_samples=samples_to_generate_per_cluster[cluster],
                                                            condition_column=str(class_column), condition_value=cls,
                                                            sec_condition_column=str(cluster_column), sec_condition_value=cluster)
                            generated_classes = np.full(generated_samples.shape[0], cls)

                            x_resampled = np.vstack((x_resampled, generated_samples))
                            y_resampled = np.hstack((y_resampled, generated_classes))

        elif self._sampling_strategy == 'unisam':
            majority_class = np.array(self._samples_per_class).argmax()
            num_majority_samples = np.max(np.array(self._samples_per_class))

            # Perform oversampling
            for cls in tqdm(range(self._n_classes), desc="ctdGAN Uniform Sampling (Ablation)    "):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._samples_per_class[cls]

                    if samples_to_generate > 1:
                        # Generate the appropriate number of samples to equalize cls with the majority class.
                        generated_samples = self.sample(num_samples=samples_to_generate,
                                                        condition_column=str(class_column), condition_value=cls)
                        generated_classes = np.full(generated_samples.shape[0], cls)

                        x_resampled = np.vstack((x_resampled, generated_samples))
                        y_resampled = np.hstack((y_resampled, generated_classes))

        elif self._sampling_strategy == 'balance-clusters':
            majority_class = np.array(self._samples_per_class).argmax()
            imb_matrix = self._clustered_transformer.imbalance_matrix_

            # Perform oversampling by performing cluster-based oversampling
            majority_samples = np.max(imb_matrix, axis=0)
            majority_classes = np.argmax(imb_matrix, axis=0)
            #print("Imbalance Matrix:\n", imb_matrix)
            #print(majority_samples)
            #print(majority_classes)

            for u in tqdm(range(self._n_clusters), desc="ctdGAN (BalClu) Sampling (Ablation)      "):
                for cls in range(self._n_classes):
                    if cls != majority_classes[u] and cls != majority_class and 3 < imb_matrix[cls][u] < imb_matrix[majority_class][u]:
                        samples_to_generate = int(imb_matrix[majority_class][u] - imb_matrix[cls][u])
                        # print("\tI will create", samples_to_generate, "samples from Class:", cls, "in Cluster:", u)

                        if samples_to_generate > 1:
                            # Generate the appropriate number of samples to equalize cls with the majority class.
                            generated_samples = self.sample(num_samples=samples_to_generate,
                                                            condition_column=str(class_column), condition_value=cls,
                                                            sec_condition_column=str(cluster_column), sec_condition_value=u)

                            if generated_samples is not None and generated_samples.shape[0] > 0:
                                # print("\t\tCreated", generated_samples.shape[0], "samples")
                                generated_classes = np.full(generated_samples.shape[0], cls)

                                x_resampled = np.vstack((x_resampled, generated_samples))
                                y_resampled = np.hstack((y_resampled, generated_classes))
                    #else:
                    #    print("\tClass:", cls, "with ", imb_matrix[cls][u], "samples ignored in Cluster:", u, "(", imb_matrix[majority_class][u], ")")
        # dictionary mode: the keys correspond to the targeted classes. The values correspond to the desired number of
        # samples for each targeted class.
        elif isinstance(self._sampling_strategy, dict):
            for cls in tqdm(self._sampling_strategy, desc="ctdGAN Sampling     "):
                # In imblearn sampling strategy stores the class distribution of the output dataset. So we have to
                # create the half number of samples, and we divide by 2.
                samples_to_generate = int(self._sampling_strategy[cls] / 2)

                # Generate the appropriate number of samples to equalize cls with the majority class.
                generated_samples = self.sample(num_samples=samples_to_generate,
                                                condition_column=str(class_column), condition_value=cls)

                if generated_samples is not None and generated_samples.shape[0] > 0:
                    # print("\t\tCreated", generated_samples.shape[0], "samples")
                    generated_classes = np.full(generated_samples.shape[0], cls)

                    x_resampled = np.vstack((x_resampled, generated_samples))
                    y_resampled = np.hstack((y_resampled, generated_classes))

        elif self._sampling_strategy == 'create-new':
            x_resampled = None
            y_resampled = None

            s = 0
            for cls in tqdm(range(self._n_classes), desc="ctdGAN Sampling     "):
                samples_to_generate = int(self._samples_per_class[cls])
                samples_to_generate_per_cluster = self._clustered_transformer.imbalance_matrix_[cls].astype(np.int32)
                print("Total samples:", samples_to_generate)
                print("Class:", cls, "- Total samples to create (per cluster):", samples_to_generate_per_cluster)

                for cluster in range(self._n_clusters):
                    if samples_to_generate_per_cluster[cluster] > 1:
                        # Generate the appropriate number of samples to equalize cls with the majority class.
                        generated_samples = self.sample(num_samples=samples_to_generate_per_cluster[cluster],
                                                        condition_column=str(class_column), condition_value=cls,
                                                        sec_condition_column=str(cluster_column),
                                                        sec_condition_value=cluster)
                        generated_classes = np.full(generated_samples.shape[0], cls)

                        if s == 0:
                            x_resampled = generated_samples
                            y_resampled = generated_classes
                            s = 1
                        else:
                            x_resampled = np.vstack((x_resampled, generated_samples))
                            y_resampled = np.hstack((y_resampled, generated_classes))

        return x_resampled, y_resampled
