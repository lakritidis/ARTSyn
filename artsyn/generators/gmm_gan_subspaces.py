# Experimental: Latent (sub)spaces for GANs

import numpy as np

import torch
import torch.nn as nn

from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

from torch.utils.data import DataLoader

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest

from DeepCoreML.generators.gan_discriminators import PackedDiscriminator
from DeepCoreML.generators.gan_generators import Generator
from DeepCoreML.generators.GAN_Synthesizer import GANSynthesizer


class GMMComponent:
    def __init__(self, label, x, y, random_state):
        self._label = label
        self._x = x
        self._y = y
        self._num_samples = self._y.shape[0]
        self._transformer = StandardScaler(with_mean=True, with_std=True)
        self._random_state = random_state

        self._mvn = None
        self._mvn_mean = None
        self._mvn_covariance_type = None
        self._mvn_covariance = None

    def display(self):
        print("\t--- Cluster ", self._label, "-------------------------------")
        print("\t\t* Num Samples: ", self._num_samples)
        print("\t\t* Prob. Distribution Covariance Type: ", self._mvn_covariance_type)
        print("\t\t* Prob. Distribution Mean: ", self._mvn_mean)
        print("\t-----------------------------------------------\n")

    def fit_transform(self):
        self._transformer.fit(self._x)
        return self._transformer.transform(self._x)

    def inverse_transform(self, x):
        return self._transformer.inverse_transform(x)

    def create_distribution(self, comp, means, covariances, covariance_type):
        self._mvn_mean = torch.from_numpy(means[comp])
        self._mvn_covariance_type = covariance_type

        if self._mvn_covariance_type == 'full':
            self._mvn_covariance = torch.from_numpy(covariances[comp])
            self._mvn = MultivariateNormal(loc=self._mvn_mean, covariance_matrix=self._mvn_covariance)

        elif self._mvn_covariance_type == 'tied':
            self._mvn_covariance = torch.from_numpy(covariances)
            self._mvn = MultivariateNormal(loc=self._mvn_mean, covariance_matrix=self._mvn_covariance)

        elif self._mvn_covariance_type == 'diag':
            self._mvn_covariance = torch.from_numpy(covariances[comp])
            self._mvn = Normal(loc=self._mvn_mean, scale=self._mvn_covariance)

        elif self._mvn_covariance_type == 'spherical':
            self._mvn_covariance = torch.full((self._mvn_mean.shape[0],), covariances[comp])
            self._mvn = Normal(loc=self._mvn_mean, scale=self._mvn_covariance)

    def get_data(self, cluster_counter):
        x_scaled = self.fit_transform()
        one_hot_encoded_labels = np.zeros((self._num_samples, cluster_counter), dtype=int)
        one_hot_encoded_labels[:, self._label] = 1

        concat = np.concatenate((x_scaled, one_hot_encoded_labels), axis=1)
        # print(self._label)
        # print(one_hot_encoded_labels)
        # print(concat)
        return concat

    def get_prob_distribution(self):
        return self._mvn

    def get_label(self):
        return self._label

    def get_mean(self):
        return self._mvn_mean

    def set_label(self, v):
        self._label = v


class ClassSubSpace:
    def __init__(self, class_label, cluster_start, max_clusters, random_state):
        self._label = class_label
        self._cluster_label_start = cluster_start

        self._num_samples = 0

        self._num_gmm_components = 0
        self._max_gmm_components = max_clusters
        self._gmm_components = []

        self._cluster_prob = []
        self._random_state = random_state

    def gmm(self, x_train, y_train):
        """Use a Gaussian Mixture Model (GMM) to cluster the samples of this subspace.
        First run a heuristic to find the optimal number of clusters and GMM covariance type.

        - `full`: each component has its own general covariance matrix.
        - `tied`: all components share the same general covariance matrix.
        - `diag`: each component has its own diagonal covariance matrix.
        - `spherical`: each component has its own single variance.
        """
        self._num_samples = y_train.shape[0]

        # Apply this heuristic to discover the best covariance_type and best number of clusters.
        n_components = range(1, self._max_gmm_components + 1)
        covariance_type = ['spherical', 'tied', 'diag', 'full']
        score = []
        for cov in covariance_type:
            for n_comp in n_components:
                if n_comp <= self._num_samples:
                    gmm = GaussianMixture(n_components=n_comp, covariance_type=cov, random_state=self._random_state)
                    gmm.fit(x_train)
                    score.append((cov, n_comp, gmm.bic(x_train)))  # <-- Compute Bayes Information Criterion (BIC)

        # Best covariance_type and best number of clusters derive from the minimum BIC.
        best = min(score, key=lambda score_tuple: score_tuple[2])
        self._num_gmm_components = best[1]       # <-- Best number of clusters (GMM components)
        best_cov_type = best[0]                  # <-- Best covariance type of GMM
        print(best)

        # Perform GMM clustering with the optimal number of clusters and the optimal covariance type.
        gmm = GaussianMixture(
            n_components=self._num_gmm_components,
            covariance_type=best_cov_type,
            random_state=self._random_state)

        gmm.fit(x_train)

        cluster_labels = gmm.predict(x_train)
        # print(cluster_labels)

        # Each GMM Component is a cluster; its samples adopt a Multivariate/Normal Gaussian Distribution.
        # Create the appropriate objects, distributions and probability vectors.
        global_cluster_label = self._cluster_label_start
        for comp in range(self._num_gmm_components):
            x_comp = x_train[cluster_labels == comp, :]
            y_comp = y_train[cluster_labels == comp]

            cluster = GMMComponent(global_cluster_label, x_comp, y_comp, self._random_state)
            cluster.create_distribution(comp, gmm.means_, gmm.covariances_, best_cov_type)
            # cluster.display()

            self._gmm_components.append(cluster)
            global_cluster_label += 1

        # Forge the cluster probability vector. Given a class, this vector shows the probability of selecting a cluster.
        self._cluster_prob = [1 / self._num_gmm_components for _ in range(self._num_gmm_components)]

    def get_data(self, cluster_counter, class_counter):
        """Retrieve the (transformed) training data related to this subspace.
        It is a concatenated representation of the training data of each cluster
        """
        # x_ret is a concatenated representation of the sample data + the cluster label
        x_ret = self._gmm_components[0].get_data(cluster_counter)

        # print("\n\n\n==== cluster 0\n\n", x_ret)
        for comp in range(1, self._num_gmm_components):
            x_ret = np.concatenate((x_ret, self._gmm_components[comp].get_data(cluster_counter)))
            # print("==== cluster ", comp, "\n\n", x_ret)

        # Concatenate with the class label
        one_hot_encoded_labels = np.zeros((self._num_samples, class_counter), dtype=int)
        one_hot_encoded_labels[:, self._label] = 1

        concat = np.concatenate((x_ret, one_hot_encoded_labels), axis=1)

        return concat

    def get_num_gmm_components(self):
        return self._num_gmm_components

    def get_cluster_probs(self):
        return self._cluster_prob

    def get_gmm_component(self, i):
        return self._gmm_components[i]

    def remove_outliers(self):
        pass

    def pca(self):
        pass


class GMM_GAN(GANSynthesizer):
    """
    GMM GAN

    Conditional GANs (cGANs) conditionally generate data from a specific class. They are trained
    by providing both the Generator and the Discriminator the input feature vectors concatenated
    with their respective one-hot-encoded class labels.

    A Packed Conditional GAN (Pac cGAN) is a cGAN that accepts input samples in packs. Pac cGAN
    uses a Packed Discriminator to prevent the model from mode collapsing.
    """

    def __init__(self, embedding_dim=128, discriminator=(128, 128), generator=(256, 256), pac=10, g_activation='tanh',
                 adaptive=False, epochs=300, batch_size=32, lr=2e-4, decay=1e-6, sampling_mode='balance',
                 max_clusters=20, projector=None, random_state=0):
        """
        Initializes a SGMM GAN.

        Args:
            embedding_dim: Size of the random sample passed to the Generator.
            discriminator: a tuple with number of neurons in each fully connected layer of the Discriminator. It
                determines the dimensionality of the output of each layer.
            generator: a tuple with number of neurons in each fully connected layer of the Generator. It
                determines the dimensionality of the output of each residual block of the Generator.
            pac: Number of samples to group together when applying the discriminator.
            adaptive: boolean value to enable/disable adaptive training.
            g_activation: The activation function of the Generator's output layer.
            epochs: Number of training epochs.
            batch_size: Number of data instances per training batch.
            lr: Learning rate parameter for the Generator/Discriminator Adam optimizers.
            decay: Weight decay parameter for the Generator/Discriminator Adam optimizers.
            max_clusters: The maximum number of clusters to create

             - `uniform`: Random sampling of class and cluster labels from a uniform distribution.
             - `prob`: Random integer sampling with probability.
             - `log-prob`: Random integer sampling with log probability.
            random_state: An integer for seeding the involved random number generators.
        """
        super().__init__(embedding_dim, discriminator, generator, pac, g_activation, adaptive, epochs, batch_size,
                         lr, decay, sampling_mode, random_state)

        self._projector = projector

        self._class_subspaces = []
        self._max_gmm_components = max_clusters
        self._num_gmm_components = 0

        # Class-Cluster probability vector: Given a class c, what is the likelihood that c exists in one of the clusters
        self._cc_prob_vector = None

    def preprocess(self, x_train, y_train):
        """
        Refine the training set with sample filtering. It invokes `prepare` to return the preprocessed data.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            A tensor with the preprocessed data.
        """
        self._n_classes = len(set(y_train))
        self._input_dim = x_train.shape[1]
        num_samples = y_train.shape[0]

        # ====== 1. Identify and remove the majority class outliers from the dataset
        # ====== Return a cleaned dataset (x_clean, y_clean)
        self._gen_samples_ratio = np.unique(y_train, return_counts=True)[1]
        '''
        majority_class = np.argmax(self._gen_samples_ratio)
        maj_samples = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] == majority_class])

        # Use an Isolation Forest to detect the outliers. The predictions array marks them with -1
        clf = IsolationForest(random_state=self._random_state).fit(maj_samples)
        predictions = clf.predict(maj_samples)

        # Copy all the minority samples to the cleaned dataset
        x_clean = np.array([x_train[s, :] for s in range(num_samples) if y_train[s] != majority_class])
        y_clean = np.array([y_train[s] for s in range(num_samples) if y_train[s] != majority_class])

        # Copy the majority samples that are not outliers to the cleaned dataset
        for s in range(maj_samples.shape[0]):
            if predictions[s] == 1:
                x_clean = np.append(x_clean, [maj_samples[s, :]], axis=0)
                y_clean = np.append(y_clean, [majority_class], axis=0)

        '''
        x_clean = x_train.copy()
        y_clean = y_train.copy()
        # (x_clean, y_clean) is the new dataset without the outliers
        # print("Clean Dataset Shape:", x_clean.shape)

        # ====== 2. Space Partitioning step
        # ====== Partition the input data space into sub-spaces and quantify the probability distributions of the
        # ====== samples contained within.
        cluster_counter = 0
        for c in range(self._n_classes):
            x_class = np.array(x_clean[y_clean == c, :])
            y_class = np.array(y_clean[y_clean == c])
            num_class_samples = y_class.shape[0]
            # print("Creating space for class", c, "num_samples=", num_class_samples)

            if num_class_samples > 1:
                sub = ClassSubSpace(c, cluster_counter, self._max_gmm_components, self._random_state)
                sub.gmm(x_class, y_class)
                self._class_subspaces.append(sub)

                cluster_counter += sub.get_num_gmm_components()

        self._num_gmm_components = cluster_counter

        # ====== 3. Create the training set
        # ====== Partition the input data space into sub-spaces and quantify the probability distributions of the
        # ====== samples contained within.
        t_data = self._class_subspaces[0].get_data(cluster_counter, self._n_classes)
        for s in range(1, self._n_classes):
            t_data = np.concatenate((t_data, self._class_subspaces[s].get_data(cluster_counter, self._n_classes)))

        # ====== 5. Finalize the training set
        training_data = torch.from_numpy(t_data).to(torch.float32)

        # print("Training Data:\n", training_data)

        return training_data

    def sample_latent_space(self, num_samples):
        """Latent space sampler.

        Samples the latent space and returns the feature vectors and the one-hot-encoded cluster and class
        representations.

        Args:
            num_samples: The number of samples to create

        Returns:
             `latent_x:` The feature vectors of the sampled data.
             `latent_y:` Concatenated vector between the one-hot-encoded cluster and the one-hot-encoded class of
             the sampled data.
        """
        # Select num_samples random classes
        latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)

        latent_x = []
        means = []
        latent_clusters = []

        # For each instance to be sampled
        for s in range(num_samples):
            latent_class_id = int(latent_classes[s])  # <-- Integer class label
            latent_class = self._class_subspaces[latent_class_id]  # <-- The corresponding subspace of class

            # Select a random cluster of the latent_class
            latent_cluster_id = np.random.choice(a=np.arange(latent_class.get_num_gmm_components(), dtype=int), size=1,
                                                 replace=True, p=latent_class.get_cluster_probs())

            # Object that represents the cluster.
            latent_cluster = latent_class.get_gmm_component(int(latent_cluster_id))
            latent_clusters.append(latent_cluster.get_label())
            means.append(latent_cluster.get_mean().tolist())

            # print("Sample", s, "-- Class", latent_class_id, '--Cluster', latent_cluster_id,
            #      "\n-- Prob Vec:", latent_class.get_cluster_probs())

            # latent_x derives by sampling the cluster's Multivariate Gaussian distribution
            lx = latent_cluster.get_prob_distribution().sample()
            latent_x.append(lx)

        # One-hot encoded clusters and classes...
        latent_clusters = torch.tensor(latent_clusters).to(torch.int64)
        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._num_gmm_components)
        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)

        # ... are concatenated together in latent_y
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)

        # Latent samples converted to Tensor
        latent_x = torch.from_numpy(np.array(latent_x, dtype="float32"))

        means = torch.tensor(means)
        return latent_x, latent_y, means

    def generator_loss(self, predicted_labels, real_labels, fake_data, m):
        # The Discriminator loss: Binary Cross Entropy between predicted and real labels
        disc_loss = nn.BCELoss(reduction='mean')(predicted_labels, real_labels)

        # Generator loss = discriminator loss + Gaussian negative log likelihood loss
        # The targets are treated as samples from Gaussian distributions with expectations and variances predicted by
        # the neural network. For a target tensor modelled as having Gaussian distribution with a tensor of expectations
        # input and a tensor of positive variances var the loss is:
        # nn.GaussianNLLLoss(full=True, reduction='mean')
        loss2 = nn.CrossEntropyLoss(reduction='mean')(fake_data, m)
        gen_loss = -loss2 + disc_loss

        return gen_loss

    def train_batch(self, real_data):
        """
        Given a batch of input data, `train_batch` updates the Discriminator and Generator weights using the respective
        optimizers and back propagation.

        Args:
            real_data: data for cGAN training: a batch of concatenated sample vectors + one-hot-encoded class vectors.
        """

        # The loss function for GAN training - applied to both the Discriminator and Generator.
        disc_loss_function = nn.BCELoss(reduction='mean')

        # If the size of the batch does not allow an organization of the input vectors in packs of size self.pac_, then
        # abort silently and return without updating the model parameters.
        num_samples = real_data.shape[0]
        if num_samples % self.pac_ != 0:
            return 0, 0

        packed_samples = num_samples // self.pac_

        # DISCRIMINATOR TRAINING
        # Create fake samples from Generator
        self.D_optimizer_.zero_grad()

        # Sample the latent feature vectors
        # For a random class, pick a cluster from the corresponding subspace. latent_y = concat(cluster, class)
        # Then sample the MVN governing this cluster to create latent_x.
        latent_x, latent_y, _ = self.sample_latent_space(packed_samples)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        # The Generator produces fake samples (their labels are 0)
        fake_x = self.G_(latent_data.to(self._device))
        fake_labels = torch.zeros((packed_samples, 1))

        # The real samples (coming from the dataset) with their one-hot-encoded classes are assigned labels eq. to 1.
        real_x = real_data[:, :self._input_dim]
        real_y = real_data[:, self._input_dim:(self._input_dim + self._num_gmm_components + self._n_classes)]
        real_labels = torch.ones((packed_samples, 1))

        # Mix (concatenate) the fake samples (from Generator) with the real ones (from the dataset).
        all_x = torch.cat((real_x.to(self._device), fake_x))
        all_y = torch.cat((real_y, latent_y)).to(self._device)
        all_labels = torch.cat((real_labels, fake_labels)).to(self._device)
        all_data = torch.cat((all_x, all_y), dim=1)

        # 7. Reshape the data to feed it to Discriminator (num_samples, dimensionality) -> (-1, pac * dimensionality)
        # The samples are packed according to self.pac parameter.
        all_data = all_data.reshape((-1, self.pac_ * (self._input_dim + self._num_gmm_components + self._n_classes)))

        # 8. Pass the mixed data to the Discriminator and train the Discriminator (update its weights with backprop).
        # The loss function quantifies the Discriminator's ability to classify a real/fake sample as real/fake.
        d_predictions = self.D_(all_data)
        disc_loss = disc_loss_function(d_predictions, all_labels)
        # print(torch.cat((d_predictions, all_labels), dim=1))
        disc_loss.backward()
        self.D_optimizer_.step()

        # GENERATOR TRAINING
        self.G_optimizer_.zero_grad()

        # Sample data from the latent spaces
        latent_x, latent_y, latent_m = self.sample_latent_space(num_samples)
        latent_data = torch.cat((latent_x, latent_y), dim=1)

        # Pass the latent data through the Generator to synthesize samples
        fake_x = self.G_(latent_data.to(self._device))

        fake_data = torch.cat((fake_x, latent_y.to(self._device)), dim=1)

        # Reshape the data to feed it to Discriminator ( (num_samples, dimensionality) -> ( -1, pac * dimensionality )
        fake_data = fake_data.reshape((-1, self.pac_ * (self._input_dim + self._n_classes + self._num_gmm_components)))

        # Compute and back propagate the Generator loss
        d_predictions = self.D_(fake_data)
        gen_loss = self.generator_loss(d_predictions, real_labels.to(self._device), fake_x, latent_m.to(self._device))
        gen_loss.backward()
        self.G_optimizer_.step()

        return disc_loss, gen_loss

    def train(self, x_train, y_train):
        """
        Conventional training process of a Cluster GAN. The Generator and the Discriminator are trained
        simultaneously in the traditional adversarial fashion by optimizing `loss_function`.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
        """
        # Modify the size of the batch to align with self.pac_
        factor = self._batch_size // self.pac_
        batch_size = factor * self.pac_

        # Prepare the data for training (Clustering, Computation of Probability Distributions, Transformations, etc.)
        training_data = self.preprocess(x_train, y_train)

        # ############################################
        # LIMITATION: The latent space must be of equal dimensionality as the data space
        self.embedding_dim_ = self._input_dim
        # ############################################

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        self.D_ = PackedDiscriminator(self.D_Arch_, input_dim=self._input_dim + self._n_classes + self._num_gmm_components,
                                      pac=self.pac_).to(self._device)
        self.G_ = (Generator(self.G_Arch_, input_dim=self.embedding_dim_ + self._n_classes + self._num_gmm_components,
                             output_dim=self._input_dim, activation=self.gen_activation_, normalize=self.batch_norm_).
                   to(self._device))

        self.D_optimizer_ = torch.optim.Adam(self.D_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))
        self.G_optimizer_ = torch.optim.Adam(self.G_.parameters(),
                                             lr=self._lr, weight_decay=self._decay, betas=(0.5, 0.9))

        disc_loss, gen_loss = 0, 0
        for epoch in range(self._epochs):
            for real_data in train_dataloader:
                if real_data.shape[0] > 1:
                    disc_loss, gen_loss = self.train_batch(real_data)

        return disc_loss, gen_loss

    def fit(self, x_train, y_train):
        """`fit` invokes the GAN training process. `fit` renders the CGAN class compatible with `imblearn`'s interface,
        allowing its usage in over-sampling/under-sampling pipelines.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.
        """
        self.train(x_train, y_train)

    def sample(self, num_samples, y=None):
        """ Create artificial samples using the GAN's Generator.

        Args:
            num_samples: The number of samples to generate.
            y: The class of the generated samples. If `None`, then samples with random classes are generated.

        Returns:
            Artificial data instances created by the Generator.
        """
        # If no specific class is required, pick classes randomly.
        if y is None:
            latent_classes = torch.from_numpy(np.random.randint(0, self._n_classes, num_samples)).to(torch.int64)
        # Otherwise, fill the classes tensor with the requested class (y) value
        else:
            latent_classes = torch.full(size=(num_samples,), fill_value=y)

        latent_classes_ohe = nn.functional.one_hot(latent_classes, num_classes=self._n_classes)

        # Select the cluster with probability self._cc_prob_vector
        latent_clusters = np.zeros(num_samples)
        latent_clusters_objs = []
        latent_x = []

        # For each sample with a specific class, pick a cluster with probability lat_subspace.get_cluster_probs().
        # Then build the latent sample vector by sampling the (Multivariate) Gaussian that describes this cluster.
        for s in range(num_samples):
            lat_subspace = self._class_subspaces[int(latent_classes[s])]

            lat_cluster = np.random.choice(a=np.arange(lat_subspace.get_num_gmm_components(), dtype=int), size=1,
                                           replace=True, p=lat_subspace.get_cluster_probs())

            latent_cluster_object = lat_subspace.get_gmm_component(int(lat_cluster))

            latent_clusters_objs.append(latent_cluster_object)
            latent_clusters[s] = latent_cluster_object.get_label()
            # print("Sample", s, "-- Class", int(latent_classes[s]), '--Cluster', int(lat_cluster), "(",
            #      latent_clusters[s], ")\n-- Prob Vec:", lat_subspace.get_cluster_probs())

            # latent_x derives by sampling the cluster's Multivariate Gaussian distribution
            lx = latent_cluster_object.get_prob_distribution().sample()
            latent_x.append(lx)

        latent_x = torch.from_numpy(np.array(latent_x, dtype="float32"))

        latent_clusters = torch.from_numpy(latent_clusters).to(torch.int64)
        latent_clusters_ohe = nn.functional.one_hot(latent_clusters, num_classes=self._num_gmm_components)
        latent_y = torch.cat((latent_clusters_ohe, latent_classes_ohe), dim=1)

        # Concatenate, copy to device, and pass to generator
        latent_data = torch.cat((latent_x, latent_y), dim=1).to(self._device)

        # Generate data from the model's Generator - The feature values of the generated samples fall into the range:
        # [-1,1]: if the activation function of the output layer of the Generator is nn.Tanh().
        # [0,1]: if the activation function of the output layer of the Generator is nn.Sigmoid().
        generated_samples = self.G_(latent_data).cpu().detach().numpy()
        # print("Generated Samples:\n", generated_samples)

        # Inverse the transformation of the generated samples
        reconstructed_samples = np.zeros((num_samples, self.embedding_dim_))
        for s in range(num_samples):
            z = generated_samples[s].reshape(1, -1)
            reconstructed_samples[s] = latent_clusters_objs[s].inverse_transform(z)

        # print("Reconstructed samples\n", reconstructed_samples)
        return reconstructed_samples

    def fit_resample(self, x_train, y_train):
        """`fit_resample` alleviates the problem of class imbalance in imbalanced datasets. In particular, this
         resampling operation equalizes the number of samples from each class by oversampling the minority class.
         The function renders clusterGAN compatible with the `imblearn`'s interface, allowing its usage in
         over-sampling/under-sampling pipelines.

        Args:
            x_train: The training data instances.
            y_train: The classes of the training data instances.

        Returns:
            x_balanced: the balanced dataset samples
            y_balanced: the classes of the samples of x_balanced
        """

        # Train the GAN with the input data
        self.train(x_train, y_train)

        # balance mode: Use the GAN to equalize the number of samples per class. This is achieved by generating
        # samples of the minority classes (i.e. we perform oversampling).
        if self._sampling_mode == 'balance':
            majority_class = np.array(self._gen_samples_ratio).argmax()
            num_majority_samples = np.max(np.array(self._gen_samples_ratio))

            x_balanced = np.copy(x_train)
            y_balanced = np.copy(y_train)

            # Perform oversampling
            for cls in range(self._n_classes):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - self._gen_samples_ratio[cls]

                    # Generate the appropriate number of samples to equalize cls with the majority class.
                    # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                    generated_samples = self.sample(samples_to_generate, cls)
                    generated_classes = np.full(samples_to_generate, cls)

                    x_balanced = np.vstack((x_balanced, generated_samples))
                    y_balanced = np.hstack((y_balanced, generated_classes))

            # Return balanced_data
            return x_balanced, y_balanced

        # synthesize_similar mode: Use the GAN to create a new dataset with identical class distribution
        # as the training set.
        elif self._sampling_mode == 'synthesize_similar':
            i = 0
            x_synthetic = None
            y_synthetic = None

            # Perform oversampling
            for cls in range(self._n_classes):
                # print("Sampling class", cls, "Create", self._gen_samples_ratio[cls], "samples")
                samples_to_generate = self._gen_samples_ratio[cls]

                generated_samples = self.sample(samples_to_generate, cls)
                generated_classes = np.full(samples_to_generate, cls)

                if i == 0:
                    x_synthetic = generated_samples
                    y_synthetic = generated_classes
                else:
                    x_synthetic = np.vstack((x_synthetic, generated_samples))
                    y_synthetic = np.hstack((y_synthetic, generated_classes))
                i += 1

            # Return balanced_data
            return x_synthetic, y_synthetic
