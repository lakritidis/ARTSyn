import numpy as np
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import ADASYN

from artsyn.generators.c_gan import cGAN
from artsyn.generators.sb_gan import sbGAN
from artsyn.generators.ct_gan import ctGAN
from artsyn.generators.ctd_gan import ctdGAN
from artsyn.generators.cbr import CBR
from artsyn.generators.ctabganplus_synthesizer import CTABGANPlusSynthesizer
from artsyn.generators.ctabgan_synthesizer import CTABGANSynthesizer
from artsyn.generators.fctgan_synthesizer import FCTGANSynthesizer
from artsyn.TabularTransformer import TabularTransformer

from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdv.single_table import CopulaGANSynthesizer


class BaseResampler:
    """Resampler base wrapper class.

    Used to wrap the ImbLearn models, C-GAN and SB-GAN"""
    def __init__(self, name, model, random_state):
        self.name_ = name
        self._model = model
        self._random_state = random_state

    def fit(self, x, y):
        self._model.fit(x, y)
        return self

    def fit_resample(self, dataset, training_set_rows, sampling_strategy='None'):
        x_train = dataset.x_[training_set_rows]
        y_train = dataset.y_[training_set_rows]

        # OneHot encode the categorical variables
        table_transformer = TabularTransformer(cont_normalizer='None', clip=False)
        table_transformer.fit(x_train, dataset.categorical_columns)
        x_transformed = table_transformer.transform(x_train)

        # Fit resample
        x_bal, y_bal = self._model.fit_resample(x_transformed, y_train)

        # Inverse the OneHot transformations to retrieve the original categorical columns
        x_bal = table_transformer.inverse_transform(x_bal)

        return x_bal, y_bal


class CTResampler(BaseResampler):
    """Resampler wrapper class - inherits from BaseResampler.

    Used to wrap ctGAN (GitHub version) and ctdGAN"""
    def __init__(self, name, model, random_state):
        super().__init__(name, model, random_state)

    def fit_resample(self, dataset, training_set_rows, sampling_strategy=None):
        x_train = dataset.x_[training_set_rows]
        y_train = dataset.y_[training_set_rows]

        x_bal, y_bal = self._model.fit_resample(x_train, y_train)

        return x_bal, y_bal


class SDVResampler(BaseResampler):
    """Resampler wrapper class - inherits from BaseResampler.

    Used to wrap the Synthetic Data Vault (SDV) models"""
    def __init__(self, name, model, random_state):
        super().__init__(name, model, random_state)

    def fit_resample(self, dataset, training_set_rows, sampling_strategy='auto'):
        x_train = dataset.x_[training_set_rows]
        y_train = dataset.y_[training_set_rows]

        # if training_set_rows is None or dataset is None:
        #    fit_data = x_train
        # else:
        #    fit_data = dataset.df_.iloc[training_set_rows, :]

        # Fit the SDV model
        fit_data = dataset.df_.iloc[training_set_rows, :]
        self._model.fit(fit_data)

        # Perform Sampling until the dataset is balanced
        gen_samples_ratio = np.unique(y_train, return_counts=True)[1]

        x_balanced = np.copy(x_train)
        y_balanced = np.copy(y_train)

        # Automatically establish balance in the dataset.
        if sampling_strategy == 'auto':
            majority_class = np.array(gen_samples_ratio).argmax()
            num_majority_samples = np.max(np.array(gen_samples_ratio))

            # Perform oversampling
            for cls in range(dataset.num_classes):
                if cls != majority_class:
                    samples_to_generate = num_majority_samples - gen_samples_ratio[cls]

                    # Generate the appropriate number of samples to equalize cls with the majority class.
                    # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                    reference_data = pd.DataFrame(data={str(dataset.class_column): [cls] * samples_to_generate})
                    generated_samples = self._model.sample_remaining_columns(
                        max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:dataset.dimensionality]

                    if generated_samples is not None and generated_samples.shape[0] > 0:
                        # print(generated_samples.shape)
                        generated_classes = np.full(generated_samples.shape[0], cls)

                        x_balanced = np.vstack((x_balanced, generated_samples))
                        y_balanced = np.hstack((y_balanced, generated_classes))

        elif isinstance(sampling_strategy, dict):
            for cls in sampling_strategy:
                # In imblearn sampling strategy stores the class distribution of the output dataset. So we have to
                # create the half number of samples, and we divide by 2.
                samples_to_generate = int(sampling_strategy[cls] / 2)

                # Generate the appropriate number of samples to equalize cls with the majority class.
                # print("\tSampling Class y:", cls, " Gen Samples ratio:", gen_samples_ratio[cls])
                reference_data = pd.DataFrame(data={str(dataset.class_column): [cls] * samples_to_generate})
                generated_samples = self._model.sample_remaining_columns(
                    max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:dataset.dimensionality]

                if generated_samples is not None and generated_samples.shape[0] > 0:
                    # print(generated_samples.shape)
                    generated_classes = np.full(samples_to_generate, cls)

                    x_balanced = np.vstack((x_balanced, generated_samples))
                    y_balanced = np.hstack((y_balanced, generated_classes))

        elif sampling_strategy == 'create-new':
            x_balanced = None
            y_balanced = None

            for cls in range(dataset.num_classes):
                # Generate as many samples, as the corresponding class cls
                samples_to_generate = int(gen_samples_ratio[cls])

                reference_data = pd.DataFrame(data={str(dataset.class_column): [cls] * samples_to_generate})
                generated_samples = self._model.sample_remaining_columns(
                    max_tries_per_batch=500, known_columns=reference_data).iloc[:, 0:dataset.dimensionality]

                if generated_samples is not None and generated_samples.shape[0] > 0:
                    # print("\t\tCreated", generated_samples.shape[0], "samples from class", cls)
                    generated_classes = np.full(generated_samples.shape[0], cls)

                    if cls == 0:
                        x_balanced = generated_samples
                        y_balanced = generated_classes
                    else:
                        x_balanced = np.vstack((x_balanced, generated_samples))
                        y_balanced = np.hstack((y_balanced, generated_classes))
                else:
                    print("Could not create samples from class", cls)
        return x_balanced, y_balanced


class TestSynthesizers:
    """
    An object that contains a collection of data over-sampling and under-sampling techniques.
    """
    def __init__(self, metadata, sampling_strategy='auto', random_state=0):
        """
        An object that contains a collection of data over-sampling and under-sampling techniques.

        Args:
            random_state: Control the randomization of the algorithm.
            sampling_strategy: how the member samplers generate/remove/replace samples.

             - If a float is passed, it corresponds to the desired ratio of the number of samples in the minority class
               over the number of samples in the majority class after resampling. float is only available for binary
               classification. An error is raised for multi-class classification.
             - If a string is passed, specify the class targeted by the resampling. The number of samples in the
               different classes will be equalized. Possible choices are:

               * 'minority': resample only the minority class;
               * 'not minority': resample all classes but the minority class;
               * 'not majority': resample all classes but the majority class;
               * 'all': resample all classes;
               * 'auto': equivalent to 'not majority'.

             - If a dictionary is passed, the keys correspond to the targeted classes. The values correspond to the
               desired number of samples for each targeted class.
             - When callable, function taking y and returns a dict. The keys correspond to the targeted classes.
               The values correspond to the desired number of samples for each class.`
        """
        self._random_state = random_state

        disc = (128, 256)
        gen = (256, 256)
        emb_dim = 128
        knn = 10
        rad = 1
        epochs = 300
        batch_size = 100
        max_clusters = 20

        # Prepare the column descriptors for the SDV models
        dp_cols = {}
        for k in metadata.columns.keys():
            dp_cols[k] = {'type': metadata.columns[k]['sdtype']}

        # Random over-sampler
        ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)

        # Synthetic Minority Oversampling Technique (SMOTE)
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

        # Borderline SMOTE
        b_smote = BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

        # SMOTE with Support Vector Machine
        svm_smote = SVMSMOTE(sampling_strategy=sampling_strategy, random_state=random_state)

        # A SMOTE variant brings balance to clusters identified by k-Means
        km_smote = KMeansSMOTE(sampling_strategy=sampling_strategy, cluster_balance_threshold='auto',
                               random_state=random_state)

        # Adaptive Synthetic Sampling (ADASYN)
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)

        # Cluster-Based Resampler (CBR)
        cbr = CBR(sampling_strategy=sampling_strategy, cluster_estimator='hac', cluster_resampler='cs', verbose=False,
                  k_neighbors=1, min_distance_factor=0.01, random_state=random_state)

        # Conditional Generative Adversarial Network (C-GAN)
        c_gan = cGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=1, epochs=epochs,
                     batch_size=batch_size, sampling_strategy=sampling_strategy, random_state=random_state)

        # Safe/Borderline Generative Adversarial Network (SB-GAN)
        sb_gan = sbGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=1, epochs=epochs,
                       batch_size=batch_size, method='knn', k=knn, r=rad, sampling_strategy=sampling_strategy,
                       random_state=random_state)

        # This ctGAN is from the GitHub implementation
        # ctgan_1 = ctGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, pac=10, epochs=epochs,
        #                batch_size=batch_size, discriminator_steps=1, log_frequency=True, verbose=False,
        #                sampling_strategy=sampling_strategy, random_state=random_state)

        # And this ctGAN is from the Synthetic Data Vault - Default Discriminator (256, 256) - Generator (256, 256)
        ctgan = CTGANSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=epochs,
                                 verbose=False)

        # Tabular Variational Autoencoder (TVAE)
        t_vae = TVAESynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=1000,
                                verbose=False)

        # Gaussian Copula
        g_cop = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False)

        # CopulaGAN
        cop_gan = CopulaGANSynthesizer(metadata, enforce_min_max_values=False, enforce_rounding=False, epochs=epochs,
                                       verbose=False)

        # CTABGAN
        ctabgan = CTABGANSynthesizer(metadata, epochs=150, random_state=random_state)

        # CTABGAN+
        ctabgan_plus = CTABGANPlusSynthesizer(metadata, epochs=150, random_state=random_state)

        # FCTGAN
        fct_gan = FCTGANSynthesizer(metadata, epochs=150, random_state=random_state)

        # CTD Generative Adversarial Network (ctdGAN)
        ctdgan_km = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                           batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='mms11',
                           cluster_method='kmeans', sampling_strategy=sampling_strategy, random_state=random_state)

        ctdgan_hac = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                            batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='mms11',
                            cluster_method='hac', sampling_strategy=sampling_strategy, random_state=random_state)

        # DCTD Generative Adversarial Network (ctdGAN)
        dctdgan_km_u = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                              batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-uni', bins='auto',
                              cluster_method='kmeans', sampling_strategy=sampling_strategy, random_state=random_state)

        dctdgan_hac_u = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                               batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-uni', bins='auto',
                               cluster_method='hac', sampling_strategy=sampling_strategy, random_state=random_state)

        dctdgan_km_q = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                              batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-q', bins='auto',
                              cluster_method='kmeans', sampling_strategy=sampling_strategy, random_state=random_state)

        dctdgan_hac_q = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                               batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-q', bins='auto',
                               cluster_method='hac', sampling_strategy=sampling_strategy, random_state=random_state)

        dctdgan_km_k = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                              batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-k', bins='auto',
                              cluster_method='kmeans', sampling_strategy=sampling_strategy, random_state=random_state)

        dctdgan_hac_k = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                               batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-k', bins='auto',
                               cluster_method='hac', sampling_strategy=sampling_strategy, random_state=random_state)

        dctdgan_km_b = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                              batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-bgm', bins='auto',
                              cluster_method='kmeans', sampling_strategy=sampling_strategy, random_state=random_state)

        dctdgan_hac_b = ctdGAN(embedding_dim=emb_dim, discriminator=disc, generator=gen, epochs=epochs,
                               batch_size=batch_size, max_clusters=max_clusters, pac=10, scaler='bins-bgm', bins=5,
                               cluster_method='hac', sampling_strategy=sampling_strategy, random_state=random_state)

        # All over-samplers.
        self.over_samplers_ = [
            # BaseResampler(name="None", model=None, random_state=random_state),
            BaseResampler(name="ROS", model=ros, random_state=random_state),
            BaseResampler(name="SMOTE", model=smote, random_state=random_state),
            BaseResampler(name="BorderSMOTE", model=b_smote, random_state=random_state),
            BaseResampler(name="SVM-SMOTE", model=svm_smote, random_state=random_state),
            BaseResampler(name="KMeans SMOTE", model=km_smote, random_state=random_state),
            BaseResampler(name="ADASYN", model=adasyn, random_state=random_state),
            BaseResampler(name="CBR", model=cbr, random_state=random_state),
            # CTResampler("ctGAN", model=ctgan_1, random_state=random_state),

            BaseResampler(name="C-GAN", model=c_gan, random_state=random_state),
            BaseResampler(name="SB-GAN", model=sb_gan, random_state=random_state),
            SDVResampler(name="CT-GAN", model=ctgan, random_state=random_state),
            SDVResampler(name="TVAE", model=t_vae, random_state=random_state),
            SDVResampler(name="GCOP", model=g_cop, random_state=random_state),
            SDVResampler(name="COP-GAN", model=cop_gan, random_state=random_state),
            SDVResampler(name="CTABGAN", model=ctabgan, random_state=random_state),
            SDVResampler(name="CTABGAN+", model=ctabgan_plus, random_state=random_state),
            SDVResampler(name="FCTBGAN", model=fct_gan, random_state=random_state),

            CTResampler("CTD-GAN-KM", model=ctdgan_km, random_state=random_state),
            CTResampler("CTD-GAN-HAC", model=ctdgan_hac, random_state=random_state),

            CTResampler("DCTD-GAN-KM-U", model=dctdgan_km_u, random_state=random_state),
            CTResampler("DCTD-GAN-KM-Q", model=dctdgan_km_q, random_state=random_state),
            CTResampler("DCTD-GAN-KM-K", model=dctdgan_km_k, random_state=random_state),
            CTResampler("DCTD-GAN-KM-BGM", model=dctdgan_km_b, random_state=random_state),
        ]

        self.over_samplers_dctd_ = [
            CTResampler("DCTD-GAN-KM-U", model=dctdgan_km_u, random_state=random_state),
            CTResampler("DCTD-GAN-KM-Q", model=dctdgan_km_q, random_state=random_state),
            CTResampler("DCTD-GAN-KM-K", model=dctdgan_km_k, random_state=random_state),
            CTResampler("DCTD-GAN-KM-BGM", model=dctdgan_km_b, random_state=random_state),

            CTResampler("DCTD-GAN-HAC-U", model=dctdgan_hac_u, random_state=random_state),
            CTResampler("DCTD-GAN-HAC-Q", model=dctdgan_hac_q, random_state=random_state),
            CTResampler("DCTD-GAN-HAC-K", model=dctdgan_hac_k, random_state=random_state),
            CTResampler("DCTD-GAN-HAC-BGM", model=dctdgan_hac_b, random_state=random_state),
        ]

        self.over_samplers_sdv_ = [
            SDVResampler(name="CTGAN", model=ctgan, random_state=random_state),
            SDVResampler(name="TVAE", model=t_vae, random_state=random_state),
            SDVResampler(name="COP-GAN", model=cop_gan, random_state=random_state),
            SDVResampler(name="GCOP", model=g_cop, random_state=random_state),
        ]

        self.num_over_samplers_ = len(self.over_samplers_)

    def clean_over_samplers(self):
        self.over_samplers_ = []
        self.num_over_samplers_ = 0

    def add_over_sampler(self, resampler):
        self.over_samplers_.append(resampler)
        self.num_over_samplers_ = len(self.over_samplers_)

    def add_resampler(self, name, model):
        """
        Add a resampler object to self.over_samplers_. In case the provided model does not support the fit_resample
        method, we extend its functionality by turning it to a BaseResampler, SDVResampler, or CTResampler object.

        Args:
            name (str): A custom descriptor for the provided model.
            model (obj): A resampler object. Supported object types include: `RandomOverSampler`, `SMOTE`,
            `BorderlineSMOTE`, `SVMSMOTE`, `KMeansSMOTE`, `ADASYN`, `CBR`, `cGAN`, `sbGAN`, `CTGANSynthesizer`,
            `TVAESynthesizer`, `GaussianCopulaSynthesizer`, `CopulaGANSynthesizer`, `CTABGANSynthesizer`, and `ctdGAN`.

        """
        if (isinstance(model, RandomOverSampler) or isinstance(model, SMOTE) or isinstance(model, BorderlineSMOTE) or
                isinstance(model, SVMSMOTE) or isinstance(model, KMeansSMOTE) or isinstance(model, ADASYN) or
                isinstance(model, CBR) or isinstance(model, cGAN) or isinstance(model, sbGAN)):

            self._add_base_resampler(name, model)

        elif (isinstance(model, CTGANSynthesizer) or isinstance(model, TVAESynthesizer) or
              isinstance(model, GaussianCopulaSynthesizer) or isinstance(model, CopulaGANSynthesizer) or
              isinstance(model, CTABGANPlusSynthesizer) or isinstance(model, CTABGANSynthesizer)):

            self._add_sdv_resampler(name, model)

        elif isinstance(model, ctdGAN) or isinstance(model, ctGAN):

            self._add_ct_resampler(name, model)

        else:
            print("The provided model object is not supported. The supported model must be one of the following:")
            print("\tRandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN, CBR,")
            print("\tcGAN, sbGAN, CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer, ")
            print("\tCopulaGANSynthesizer, CTABGANSynthesizer, ctdGAN")

    def _add_base_resampler(self, name, model):
        resampler = BaseResampler(name=name, model=model, random_state=self._random_state)
        self.over_samplers_.append(resampler)
        self.num_over_samplers_ = len(self.over_samplers_)

    def _add_sdv_resampler(self, name, model):
        resampler = SDVResampler(name=name, model=model, random_state=self._random_state)
        self.over_samplers_.append(resampler)
        self.num_over_samplers_ = len(self.over_samplers_)

    def _add_ct_resampler(self, name, model):
        resampler = CTResampler(name=name, model=model, random_state=self._random_state)
        self.over_samplers_.append(resampler)
        self.num_over_samplers_ = len(self.over_samplers_)
