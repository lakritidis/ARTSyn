# SB-GAN: Safe-Borderline Generative Adversarial Net
# Introduced in the following paper:
#
# L. Akritidis, A. Fevgas, M. Alamaniotis, P. Bozanis, "Conditional Data Synthesis with Deep Generative Models for
# Imbalanced Dataset Oversampling", In Proceedings of the 35th IEEE International Conference on Tools with Artificial
# Intelligence (ICTAI), pp. 444-451, 2023.

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.neighbors import KDTree

from tads.TabularTransformer import TabularTransformer
from tads.generators.gan_discriminators import PackedDiscriminator
from tads.generators.gan_generators import Generator
from tads.generators.GAN_Synthesizer import GANSynthesizer


class sbGAN(GANSynthesizer):
    """
    Safe-Borderline GAN

    Conditional GANs (cGANs) conditionally generate data from a specific class. They are trained
    by providing both the Generator and the Discriminator the input feature vectors concatenated
    with their respective one-hot-encoded class labels.

    A Packed Conditional GAN (Pac cGAN) is a cGAN that accepts input samples in packs. Pac cGAN
    uses a Packed Discriminator to prevent the model from mode collapsing.
    """

    def __init__(self, embedding_dim=128, discriminator=(128, 128), generator=(256, 256), epochs=300, batch_size=32,
                 pac=10, lr=2e-4, decay=1e-6, g_activation='tanh', sampling_strategy='auto', method='knn', k=5, r=10,
                 random_state=0):
        """
        Initializes a Safe-Borderline Conditional GAN.

        Args:
            embedding_dim: Size of the random sample passed to the Generator.
            discriminator: a tuple with number of neurons in each fully connected layer of the Discriminator. It
                determines the dimensionality of the output of each layer.
            generator: a tuple with number of neurons in each fully connected layer of the Generator. It
                determines the dimensionality of the output of each residual block of the Generator.
            pac: Number of samples to group together when applying the discriminator.
            g_activation: The activation function of the Generator's output layer.
            epochs: Number of training epochs.
            batch_size: Number of data instances per training batch.
            lr: Learning rate parameter for the Generator/Discriminator Adam optimizers.
            decay: Weight decay parameter for the Generator/Discriminator Adam optimizers.
            method: 'knn': k-nearest neighbors / 'rad': neighbors inside a surrounding hypersphere.
            k: The number of nearest neighbors to retrieve; applied when `method='knn'`.
            r: The radius of the hypersphere that includes the neighboring samples; applied when `method='rad'`.
            sampling_strategy: How the model generates data.
            random_state (int): Seed the random number generators. Use the same value for reproducible results.
        """
        super().__init__("SB-GAN", embedding_dim, discriminator, generator, pac, epochs, batch_size,
                         lr, lr, decay, decay, sampling_strategy, random_state)

        self.gen_activation_ = g_activation
        self._method = method
        self._n_neighbors = k
        self._radius = r

    def select_prepare(self, x_train, y_train):
        """
        Refine the training set with sample filtering. It invokes `prepare` to return the preprocessed data.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            A tensor with the preprocessed data.
        """
        num_samples = x_train.shape[0]

        # Array with the point labels.
        pts_types = ['NotSet'] * num_samples
        x_sample, y_sample = [], []

        # Build an auxiliary KD-Tree accelerate spatial queries.
        kd_tree = KDTree(x_train, metric='euclidean', leaf_size=40)

        # Nearest-Neighbors method.
        if self._method == 'knn':
            # Query the KD-Tree to find the nearest neighbors. indices contains the indices of the nearest neighbors
            # in the original dataset -> a row indices[r] contains the 5 nearest neighbors of x_train[r].
            _, indices = kd_tree.query(x_train, k=self._n_neighbors)

        elif self._method == 'rad':
            # Query the KD-Tree to find the neighbors-within-hypersphere of radius=radius. indices contains the indices
            # of the neighbors-within-hypersphere in the original dataset -> a row indices[r] contains the
            # neighbors-within-hypersphere of x_train[r].
            indices = kd_tree.query_radius(x_train, r=self._radius)

        else:
            print("method should be 'knn' or 'rad'; returning the input dataset")
            return self.prepare(x_train, y_train)

        # For each sample in the dataset
        for m in range(num_samples):
            pts_with_same_class = 0

            # Examine its nearest neighbors and assign a label: core/border/outlier/isolated
            num_neighbors = len(indices[m])
            for k in range(num_neighbors):
                nn_idx = indices[m][k]
                if y_train[nn_idx] == y_train[m]:
                    pts_with_same_class += 1

            if num_neighbors == 1:
                pts_types[m] = 'Isolated'
            else:
                t_high = 1.0 * num_neighbors
                t_low = 0.2 * num_neighbors

                if pts_with_same_class >= t_high:
                    pts_types[m] = 'Core'
                    x_sample.append(x_train[m])
                    y_sample.append(y_train[m])
                elif t_high > pts_with_same_class > t_low:
                    pts_types[m] = 'Border'
                    x_sample.append(x_train[m])
                    y_sample.append(y_train[m])
                else:
                    pts_types[m] = 'Outlier'

            # print("Sample", m, ":", pts_types[m], "- Neighbors indices:", indices[m], "- Neighbors classes:",
            #      [y_train[indices[m][k]] for k in range(num_neighbors)])

        x_train = np.array(x_sample)
        y_train = np.array(y_sample)

        return self.prepare(x_train, y_train)

    def train_batch(self, real_data):
        """
        Given a batch of input data, `train_batch` updates the Discriminator and Generator weights using the respective
        optimizers and back propagation.

        Args:
            real_data: data for cGAN training: a batch of concatenated sample vectors + one-hot-encoded class vectors.

        Returns:
            disc_loss: The Discriminator's loss.
            gen_loss: The Generator's loss.
        """

        # The loss function for GAN training - applied to both the Discriminator and Generator.
        loss_function = nn.BCELoss()

        # If the size of the batch does not allow an organization of the input vectors in packs of size self._pac, then
        # abort silently and return without updating the model parameters.
        num_samples = real_data.shape[0]
        if num_samples % self._pac != 0:
            print("pac error")
            return 0, 0

        # DISCRIMINATOR TRAINING
        # Create fake samples from Generator
        self.D_optimizer_.zero_grad()

        # 1. Randomly take samples from a normal distribution
        # 2. Assign one-hot-encoded random classes
        # 3. Pass the fake data (samples + classes) to the Generator
        latent_x = torch.randn((num_samples, self._embedding_dim))
        latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
        latent_y = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        # 4. The Generator produces fake samples (their labels are 0)
        fake_x = self.G_(latent_data.to(self._device))
        fake_labels = torch.zeros((num_samples // self._pac, 1))

        # 5. The real samples (coming from the dataset) with their one-hot-encoded classes are assigned labels eq. to 1.
        real_x = real_data[:, 0:self._input_dim]
        real_y = real_data[:, self._input_dim:(self._input_dim + self._n_classes)]
        real_labels = torch.ones((num_samples // self._pac, 1))

        # 6. Mix (concatenate) the fake samples (from Generator) with the real ones (from the dataset).
        all_x = torch.cat((real_x.to(self._device), fake_x))
        all_y = torch.cat((real_y, latent_y)).to(self._device)
        all_labels = torch.cat((real_labels, fake_labels)).to(self._device)
        all_data = torch.cat((all_x, all_y), dim=1)

        # 7. Pass the mixed data to the Discriminator and train the Discriminator (update its weights with backprop).
        # The loss function quantifies the Discriminator's ability to classify a real/fake sample as real/fake.
        d_predictions = self.D_(all_data)
        disc_loss = loss_function(d_predictions, all_labels)
        disc_loss.backward()
        self.D_optimizer_.step()

        # GENERATOR TRAINING
        self.G_optimizer_.zero_grad()

        latent_x = torch.randn((num_samples, self._embedding_dim))
        latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
        latent_y = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        fake_x = self.G_(latent_data.to(self._device))

        all_data = torch.cat((fake_x, latent_y.to(self._device)), dim=1)

        d_predictions = self.D_(all_data)

        gen_loss = loss_function(d_predictions, real_labels.to(self._device))
        gen_loss.backward()
        self.G_optimizer_.step()

        return disc_loss, gen_loss

    def train(self, x_train, y_train):
        """
        Conventional training process of a Packed sbGAN. The Generator and the Discriminator are trained
        simultaneously in the traditional adversarial fashion by optimizing `loss_function`.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
        """

        # Modify the size of the batch to align with self._pac
        factor = self._batch_size // self._pac
        batch_size = factor * self._pac

        self._transformer = TabularTransformer(cont_normalizer='stds')
        self._transformer.fit(x_train)
        x_train = self._transformer.transform(x_train)

        # select_prepare: implemented in GAN_Synthesizer.py
        training_data = self.select_prepare(x_train, y_train)

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=self._input_dim + self._n_classes,
                                      pac=self._pac).to(self._device)
        self.G_ = Generator(self.G_Arch_, input_dim=self._embedding_dim + self._n_classes, output_dim=self._input_dim,
                            activation=self.gen_activation_, normalize=self._batch_norm).to(self._device)

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._disc_lr, weight_decay=self._disc_decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._gen_lr, weight_decay=self._gen_decay, betas=(0.5, 0.9))

        disc_loss, gen_loss = 0, 0
        for _ in tqdm(range(self._epochs), desc="SB-GAN Training     "):
            for real_data in train_dataloader:
                if real_data.shape[0] > 1:
                    disc_loss, gen_loss = self.train_batch(real_data)

                # if epoch % 10 == 0 and n >= x_train.shape[0] // batch_size:
                #    print(f"Epoch: {epoch} Loss D.: {disc_loss} Loss G.: {gen_loss}")

        return disc_loss, gen_loss

    def fit(self, x_train, y_train):
        """`fit` invokes the sbGAN training process. `fit` renders sbGAN compatible with `imblearn`'s interface,
        allowing its usage in over-sampling/under-sampling pipelines.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
        """
        self.train(x_train, y_train)

    # Use GAN's Generator to create artificial samples i) either from a specific class, ii) or from a random class.
    def sample(self, num_samples, y=None):
        """ Create artificial samples using the sbGAN's Generator.

        Args:
            num_samples: The number of samples to generate.
            y: The class of the generated samples. If `None`, then samples with random classes are generated.

        Returns:
            Artificial data instances created by the Generator.
        """
        if y is None:
            latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
            latent_y = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)
        else:
            # print("requested class:", y, "Num samples:", num_samples, "Num classes:", self._n_classes)
            latent_y = nn.functional.one_hot(torch.full(size=(num_samples,), fill_value=y), num_classes=self._n_classes)

        latent_x = torch.randn((num_samples, self._embedding_dim))

        # concatenate, copy to device, and pass to generator
        latent_data = torch.cat((latent_x, latent_y), dim=1).to(self._device)

        # Generate data from the model's Generator - The feature values of the generated samples fall into the range:
        # [-1,1]: if the activation function of the output layer of the Generator is nn.Tanh().
        # [0,1]: if the activation function of the output layer of the Generator is nn.Sigmoid().

        generated_samples = self.G_(latent_data).cpu().detach().numpy()
        # print("Generated Samples:\n", generated_samples)

        reconstructed_samples = self._transformer.inverse_transform(generated_samples)
        # print("Reconstructed samples\n", reconstructed_samples)
        return reconstructed_samples

    def fit_resample(self, x_train, y_train):
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. The function renders sbGAN
        compatible with the `imblearn`'s interface, allowing its usage in over-sampling/under-sampling pipelines.

        In the `fit` part, the input dataset is used to train sbGAN. In the `resample` part, sbGAN is employed to
        generate synthetic data according to the value of `self._sampling_strategy`:

        - 'auto': the model balances the dataset by oversampling the minority classes.
        - dict: a dictionary that indicates the number of samples to be generated from each class

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            x_resampled: The training data instances + the generated data instances.
            y_resampled: The classes of the training data instances + the classes of the generated data instances.
        """

        # Train the GAN with the input data
        self.train(x_train, y_train)

        x_resampled = np.copy(x_train)
        y_resampled = np.copy(y_train)

        # auto mode: Use sbGAN to equalize the number of samples per class. This is achieved by generating samples
        # of the minority classes (i.e. we perform oversampling).
        if self._sampling_strategy == 'auto':
            majority_class = np.array(self._gen_samples_ratio).argmax()
            num_majority_samples = np.max(np.array(self._gen_samples_ratio))

            # Perform oversampling
            for cls in tqdm(range(self._n_classes), desc="SB-GAN Sampling     "):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._gen_samples_ratio[cls]

                    if samples_to_generate > 1:
                        # Generate the appropriate number of samples to equalize cls with the majority class.
                        # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                        generated_samples = self.sample(samples_to_generate, cls)
                        generated_classes = np.full(samples_to_generate, cls)

                        x_resampled = np.vstack((x_resampled, generated_samples))
                        y_resampled = np.hstack((y_resampled, generated_classes))

        # dictionary mode: the keys correspond to the targeted classes. The values correspond to the desired number of
        # samples for each targeted class.
        elif isinstance(self._sampling_strategy, dict):
            for cls in tqdm(self._sampling_strategy, desc="SB-GAN Sampling     "):
                # In imblearn sampling strategy stores the class distribution of the output dataset. So we have to
                # create the half number of samples, and we divide by 2.
                samples_to_generate = int(self._sampling_strategy[cls] / 2)

                # Generate the appropriate number of samples to equalize cls with the majority class.
                # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                generated_samples = self.sample(samples_to_generate, cls)
                generated_classes = np.full(samples_to_generate, cls)

                x_resampled = np.vstack((x_resampled, generated_samples))
                y_resampled = np.hstack((y_resampled, generated_classes))

        elif self._sampling_strategy == 'create-new':
            x_resampled = None
            y_resampled = None

            s = 0
            for cls in tqdm(range(self._n_classes), desc="SB-GAN Sampling     "):
                # Generate as many samples, as the corresponding class cls
                samples_to_generate = int(self._gen_samples_ratio[cls])
                generated_samples = self.sample(num_samples=samples_to_generate, y=cls)

                if generated_samples is not None and generated_samples.shape[0] > 0:
                    # print("\t\tCreated", generated_samples.shape[0], "samples")
                    generated_classes = np.full(generated_samples.shape[0], cls)

                    if s == 0:
                        x_resampled = generated_samples
                        y_resampled = generated_classes
                        s = 1
                    else:
                        x_resampled = np.vstack((x_resampled, generated_samples))
                        y_resampled = np.hstack((y_resampled, generated_classes))
        return x_resampled, y_resampled
