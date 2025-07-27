# cGAN: Conditional Generative Adversarial Net
# Introduced in the following paper:
#
# M. Mirza, S. Osindero, "Conditional Generative Adversarial Nets", arXiv preprint arXiv:1411.1784, 2014.

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.metrics import accuracy_score

from artsyn.TabularTransformer import TabularTransformer
from artsyn.generators.gan_discriminators import PackedDiscriminator
from artsyn.generators.gan_generators import Generator
from artsyn.generators.GAN_Synthesizer import GANSynthesizer


class cGAN(GANSynthesizer):
    """Conditional GAN (CGAN)

    Conditional GANs conditionally generate data from a specific class.
    """

    def __init__(self, embedding_dim=128, discriminator=(128, 128), generator=(256, 256), epochs=300, batch_size=32,
                 pac=10, lr=2e-4, decay=1e-6, g_activation='tanh', sampling_strategy='auto', random_state=0):

        """CGAN Initializer

        Args:
            embedding_dim (int): Size of the random sample passed to the Generator.
            discriminator (tuple): a tuple with number of neurons for each fully connected layer of the model's Critic.
                It determines the dimensionality of the output of each layer.
            generator (tuple): a tuple with number of neurons for each fully connected layer of the model's Generator.
                It determines the dimensionality of the output of each residual block of the Generator.
            epochs (int): The number of training epochs.
            batch_size (int): The number of data instances per training batch.
            pac (int): The number of samples to group together when applying the Critic.
            lr (real): The value of the learning rate parameter for the Generator/Critic Adam optimizers.
            decay (real): The value of the weight decay parameter for the Generator/Critic Adam optimizers.
            g_activation (string): The activation function for the Generator's output layer.
            sampling_strategy (string or dictionary): How the algorithm generates samples:

              * 'auto': the model balances the dataset by oversampling the minority classes.
              * dict: a dictionary that indicates the number of samples to be generated from each class.
            random_state (int): Seed the random number generators. Use the same value for reproducible results.
        """
        super().__init__("CGAN", embedding_dim, discriminator, generator, pac, epochs, batch_size,
                         lr, lr, decay, decay, sampling_strategy, random_state)

        self.gen_activation_ = g_activation
        self.test_classifier_ = None

    def train_batch(self, real_data):
        """
        Given a batch of input data, `train_batch` updates the Discriminator and Generator weights using the respective
        optimizers and back propagation.

        Args:
            real_data (2D NumPy array): data for cGAN training: a batch of concatenated
                 sample vectors + one-hot-encoded class vectors.

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
        Conventional training process of a Packed cGAN. The Generator and the Discriminator are trained simultaneously
        in the traditional adversarial fashion by optimizing `loss_function`.

        Args:
            x_train (2D NumPy array): The training data instances.
            y_train (1D NumPy array): The classes of the training data instances.
        """

        # Modify the size of the batch to align with self._pac
        self._transformer = TabularTransformer(cont_normalizer='stds')
        self._transformer.fit(x_train)
        x_train = self._transformer.transform(x_train)

        training_data = self.prepare(x_train, y_train)
        train_dataloader = DataLoader(training_data, batch_size=self._batch_size, shuffle=True)

        self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=self._input_dim + self._n_classes,
                                      pac=self._pac).to(self._device)
        self.G_ = Generator(self.G_Arch_, input_dim=self._embedding_dim + self._n_classes, output_dim=self._input_dim,
                            activation=self.gen_activation_, normalize=self._batch_norm).to(self._device)

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._disc_lr, weight_decay=self._disc_decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._gen_lr, weight_decay=self._gen_decay, betas=(0.5, 0.9))

        disc_loss, gen_loss = 0, 0

        for _ in tqdm(range(self._epochs), desc="Cond GAN Training   "):
            for real_data in train_dataloader:
                if real_data.shape[0] > 1:
                    disc_loss, gen_loss = self.train_batch(real_data)

                # if epoch % 10 == 0 and n >= x_train.shape[0] // batch_size:
                #    print(f"Epoch: {epoch} Loss D.: {disc_loss} Loss G.: {gen_loss}")

        return disc_loss, gen_loss

    def adaptive_train(self, x_train, y_train, clf, gen_samples_ratio=None):
        """ Adaptive cGAN training (experimental)

        Adaptive training by evaluating the quality of generated data during each epoch. Adaptive training is
        self-terminated when max training accuracy (of a classifier `clf`) is achieved.

        Args:
            x_train (2D NumPy array): The training data instances.
            y_train (1D NumPy array): The classes of the training data instances.
            clf (object): A classifier that has been previously trained on the training set. Its performance is measured
                at each training epoch.
            gen_samples_ratio (tuple/list): Denotes the number of samples to be generated from each class.
        """

        # Use KL Divergence to measure the distance between the real and generated data.
        kl_loss = nn.KLDivLoss(reduction="sum", log_target=True)

        factor = self._batch_size // self._pac
        batch_size = factor * self._pac

        self._transformer = TabularTransformer(cont_normalizer='ss')
        self._transformer.fit(x_train)
        x_train = self._transformer.transform(x_train)

        training_data = self.prepare(x_train, y_train)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=self._input_dim + self._n_classes,
                                      pac=self._pac).to(self._device)
        self.G_ = Generator(self.G_Arch_, input_dim=self._input_dim + self._n_classes, output_dim=self._input_dim,
                            activation=self.gen_activation_, normalize=self._batch_norm).to(self._device)

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._disc_lr, weight_decay=self._disc_decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._gen_lr, weight_decay=self._gen_decay, betas=(0.5, 0.9))

        if gen_samples_ratio is None:
            gen_samples_ratio = self._gen_samples_ratio

        generated_data = [[None for _ in range(self._n_classes)] for _ in range(self._epochs)]
        mean_met = np.zeros(self._epochs)
        mean_kld = np.zeros(self._epochs)

        # Begin training in epochs
        disc_loss, gen_loss = 0, 0
        for epoch in range(self._epochs):

            # Train the GAN in batches
            for real_data in train_dataloader:
                temp_disc_loss, temp_gen_loss = self.train_batch(real_data)
                if temp_gen_loss > 0:
                    gen_loss = temp_gen_loss
                if temp_disc_loss > 0:
                    disc_loss = temp_disc_loss

            # After the GAN has been trained on the entire dataset (for the running epoch), perform sampling with the
            # Generator (of the running epoch)
            sum_acc, sum_kld = 0, 0
            for y in range(self._n_classes):
                # print("\tSampling Class y:", y, " Gen Samples ratio:", gen_samples_ratio[y])
                generated_data[epoch][y] = self.sample(gen_samples_ratio[y], y)

                # Convert the real data of this batch to a log-probability distribution
                real_x_log_prob = torch.log(nn.Softmax(dim=0)(self._samples_per_class[y]))

                # Convert the generated data of this batch to a log-probability distribution
                gen_x_log_prob = torch.log(nn.Softmax(dim=0)(generated_data[epoch][y]))

                # Compute the KL Divergence between the real and generated data
                kld = kl_loss(real_x_log_prob, gen_x_log_prob)

                # Move the generated data to CPU and compute the classifier's performance
                generated_data[epoch][y] = generated_data[epoch][y].cpu().detach().numpy()

                if self.test_classifier_ is not None:
                    y_predicted = clf.predict(generated_data[epoch][y])
                    y_ref = np.empty(y_predicted.shape[0])
                    y_ref.fill(y)

                    acc = accuracy_score(y_ref, y_predicted)
                    sum_acc += acc

                    # print("\t\tModel Accuracy for this class:", acc, " - kl div=", kld)

                sum_kld += kld

                # print(f"Epoch: {epoch+1} \tClass: {y} \t Accuracy: {acc}")

            # if (epoch+1) % 10 == 0:
            mean_met[epoch] = sum_acc / self._n_classes
            mean_kld[epoch] = sum_kld / self._n_classes

            print("Epoch %4d \t Loss D.=%5.4f \t Loss G.=%5.4f \t Mean Acc=%5.4f \t Mean KLD=%5.4f" %
                  (epoch + 1, disc_loss, gen_loss, float(mean_met[epoch]), float(mean_kld[epoch])))

        return generated_data, (mean_met, mean_kld)

    def fit(self, x_train, y_train):
        """`fit` invokes the GAN training process. `fit` renders cGAN compatible with `imblearn`'s interface,
        allowing its usage in over-sampling/under-sampling pipelines.

        Args:
            x_train (2D NumPy array): The training data instances.
            y_train (1D NumPy array): The classes of the training data instances.
        """
        self.train(x_train, y_train)

    # Use GAN's Generator to create artificial samples i) either from a specific class, ii) or from a random class.
    def sample(self, num_samples, y=None):
        """ Create artificial samples using the cGAN's Generator.

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
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. The function renders cGAN
        compatible with the `imblearn`'s interface, allowing its usage in over-sampling/under-sampling pipelines.

        In the `fit` part, the input dataset is used to train cGAN. In the `resample` part, cGAN is employed to
        generate synthetic data according to the value of `self._sampling_strategy`:

        - 'auto': the model balances the dataset by oversampling the minority classes.
        - dict: a dictionary that indicates the number of samples to be generated from each class.

        Args:
            x_train (2D NumPy array): The training data instances.
            y_train (1D NumPy array): The classes of the training data instances.

        Returns:
            x_resampled: The training data instances + the generated data instances.
            y_resampled: The classes of the training data instances + the classes of the generated data instances.
        """

        # Train the GAN with the input data
        self.train(x_train, y_train)

        x_resampled = np.copy(x_train)
        y_resampled = np.copy(y_train)

        # auto mode: Use the Conditional GAN to equalize the number of samples per class. This is achieved by generating
        # samples of the minority classes (i.e. we perform oversampling).
        if self._sampling_strategy == 'auto':
            majority_class = np.array(self._gen_samples_ratio).argmax()
            num_majority_samples = np.max(np.array(self._gen_samples_ratio))

            # Perform oversampling
            for cls in tqdm(range(self._n_classes), desc="Cond GAN Sampling   "):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._gen_samples_ratio[cls]

                    # Generate the appropriate number of samples to equalize cls with the majority class.
                    # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                    generated_samples = self.sample(samples_to_generate, cls)
                    generated_classes = np.full(samples_to_generate, cls)

                    x_resampled = np.vstack((x_resampled, generated_samples))
                    y_resampled = np.hstack((y_resampled, generated_classes))

        # dictionary mode: the keys correspond to the targeted classes. The values correspond to the desired number of
        # samples for each targeted class.
        elif isinstance(self._sampling_strategy, dict):
            for cls in tqdm(self._sampling_strategy, desc="Cond GAN Sampling   "):
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

            # print(self._gen_samples_ratio)
            for cls in tqdm(range(self._n_classes), desc="Cond GAN Sampling   "):
                # Generate as many samples, as the corresponding class cls
                samples_to_generate = int(self._gen_samples_ratio[cls])
                generated_samples = self.sample(num_samples=samples_to_generate, y=cls)

                # print("Must create", samples_to_generate, "from class", cls, " - Created", generated_samples.shape[0])

                if generated_samples is not None and generated_samples.shape[0] > 0:
                    # print("\t\tCreated", generated_samples.shape[0], "samples")
                    generated_classes = np.full(generated_samples.shape[0], cls)

                    if cls == 0:
                        x_resampled = generated_samples
                        y_resampled = generated_classes
                    else:
                        x_resampled = np.vstack((x_resampled, generated_samples))
                        y_resampled = np.hstack((y_resampled, generated_classes))

                    # print(x_resampled.shape)
        return x_resampled, y_resampled
