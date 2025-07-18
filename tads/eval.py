import numpy as np
import time
import inspect
from tqdm import tqdm

import paths

from tads.generators.sb_gan import sbGAN
from tads.generators.c_gan import cGAN
from tads.generators.ct_gan import ctGAN
from tads.generators.ctd_gan import ctdGAN
from tads.generators.ctd_discretizer import ctdDiscretizer

from tads.TabularTransformer import TabularTransformer
from tads.TabularDataset import TabularDataset
from tads.Resamplers import TestSynthesizers
from tads.Tools import set_random_states, get_random_states, reset_random_states
from tads.ResultHandler import ResultHandler
from tads.Classifiers import Classifiers

from imblearn.pipeline import make_pipeline
from imblearn.metrics import sensitivity_score, specificity_score

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sdv.metadata import SingleTableMetadata

np.set_printoptions(precision=2)


def test_model(model, dataset, seed):
    set_random_states(seed)

    """Create, train and test individual generators.
    """
    dset = TabularDataset(name='test', class_column=dataset['class_col'],
                          categorical_columns=dataset['categorical_cols'],  random_state=seed)
    dset.load_from_csv(path=dataset['path'])
    dset.display_params()

    x = dset.x_
    y = dset.y_

    t_s = time.time()

    pac = 10
    batch_size = 100

    epochs = 10

    if model == "SBGAN":
        gan = sbGAN(discriminator=(128, 128), generator=(128, 256, 128), epochs=epochs, batch_size=batch_size,
                    pac=pac, method='knn', k=5, random_state=seed)
    elif model == "CGAN":
        gan = cGAN(discriminator=(128, 128), generator=(128, 256, 128), epochs=epochs, batch_size=batch_size,
                   pac=pac, random_state=seed)
    elif model == "CTGAN":
        gan = ctGAN(discriminator=(256, 256), generator=(256, 256), epochs=epochs, batch_size=batch_size,
                    pac=pac, random_state=seed)
    elif model == "CTDGAN":
        gan = ctdGAN(discriminator=(256, 256), generator=(256, 256), epochs=epochs, batch_size=batch_size,
                     pac=pac, embedding_dim=128, max_clusters=20, cluster_method='kmeans', scaler='mms11',
                     sampling_strategy='create-new', random_state=seed)
    elif model == "CTDGAN-R":
        gan = ctdGAN(discriminator=(256, 256), generator=(256, 256), epochs=epochs, batch_size=batch_size,
                     pac=pac, embedding_dim=128,  max_clusters=20, cluster_method='gmm', scaler='stds',
                     sampling_strategy='balance-clusters', random_state=seed)
    elif model == "D-CTDGAN":
        gan = ctdGAN(discriminator=(256, 256), generator=(256, 256), epochs=epochs, batch_size=batch_size,
                     pac=pac, embedding_dim=128, max_clusters=20, cluster_method='kmeans', scaler='bins-uni',
                     sampling_strategy='create-new', random_state=seed)
    else:
        print("No model specified")
        exit()

    balanced_data = gan.fit_resample(x, y, categorical_columns=dataset['categorical_cols'])
    # balanced_data = gan.fit_resample(x, y)
    print("Balanced Data shape:", balanced_data[0].shape)
    # print(balanced_data[0])
    print("Finished in", time.time() - t_s, "sec")


# Evaluate the ability of a resampling method to improve classification performance. This procedure employs k-fold
# cross validation without a Pipeline. Instead, we manually control what takes place in each cross validation fold.
def eval_resampling(datasets, num_folds=5, transformer=None, random_state=0):
    """Evaluate the ability of a resampling method to improve classification performance. This procedure employs k-fold
    cross validation without a Pipeline. Instead, we manually control what takes place in each cross validation fold.
    During each fold, the following steps take place:

    1. The resampler is trained,
    2. The resampler generates artificial data instances to restore balance among the classes,
    3. The balanced data are normalized (optional),
    4. A set of classifiers are trained and tested on the balanced data.

    Args:
        datasets (dict): The datasets to be used for evaluation.
        num_folds (int): The number of cross validation folds.
        transformer (str): Determines if/how the balanced data will be normalized.
        random_state: Controls random number generation. Set this to a fixed integer to get reproducible results.

    """
    # Truncate the error log
    open(paths.resampling_path_performance + paths.resampling_filename + "ErrorLog.txt", 'w').close()

    # Initialize the random number generators
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    # Determine the evaluation measures to be used - Fit time is not included here.
    scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }

    n_dataset, n_fold = 0, 0

    # For each dataset
    for key in datasets.keys():
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        n_dataset += 1
        # if num_dataset > 1:
        #     break

        # Load the dataset from the input CSV file
        ds = datasets[key]

        dataset = TabularDataset(key, class_column=ds['class_col'], categorical_columns=ds['categorical_cols'],
                                 random_state=random_state)
        dataset.load_from_csv(path=ds['path'])
        performance_list = []

        print("\n===================================================================================================")
        print("Resampling effectiveness experiment")
        dataset.display_params()

        # A SingleTableMetadata() object is required by the SDV models
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(dataset.df_)
        k = list(metadata.columns.keys())[len(metadata.columns.keys()) - 1]
        for k in metadata.columns.keys():
            if k in dataset.categorical_columns:
                metadata.columns[k] = {'sdtype': 'categorical'}
            else:
                metadata.columns[k] = {'sdtype': 'numerical'}
        # The last column becomes categorical - This structure is required by the SDV models.
        metadata.columns[k] = {'sdtype': 'categorical'}

        # Apply k-fold cross validation
        skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
        n_fold = 0

        # For each fold
        for train_idx, test_idx in skf.split(dataset.x_, dataset.y_):
            n_fold += 1
            print("\tFold: ", n_fold)

            x_test = dataset.x_[test_idx]
            y_test = dataset.y_[test_idx]

            # Initialize a new set of data samplers
            synthesizers = TestSynthesizers(metadata, sampling_strategy='auto', random_state=random_state)

            # For each sampler, fit and resample
            num_synthesizer = 0
            for synthesizer in synthesizers.over_samplers_:
                num_synthesizer += 1
                t_s = time.time()

                reset_random_states(np_random_state, torch_random_state, cuda_random_state)
                print("\t\tSynthesizer: ", synthesizer.name_)

                # Generate synthetic data with the sampler.
                if synthesizer.name_ == 'None':
                    x_balanced = dataset.x_[train_idx]
                    y_balanced = dataset.y_[train_idx]
                else:
                    try:
                        x_balanced, y_balanced = synthesizer.fit_resample(dataset=dataset, training_set_rows=train_idx,
                                                                          sampling_strategy='auto')
                    except (ValueError, RuntimeError) as e:
                        print("\t\t\t == Exception Caught: '", e, "' ==")
                        fh = open(paths.resampling_path_performance + paths.resampling_filename + "ErrorLog.txt", "a")
                        fh.write(" == Exception Caught: '" + str(e) + "' ==\n")
                        fh.write("\tSynthesizer: " + synthesizer.name_ + "\n")
                        fh.write("\tDataset: " + dataset.get_name() + "\n")
                        fh.write("\tFold: " + str(n_fold) + "\n")
                        fh.write("\tSeed: " + str(random_state) + "\n\n")
                        fh.close()
                        continue

                oversampling_duration = time.time() - t_s

                # Normalize data before feeding it to the classifiers
                if transformer == 'standardizer':
                    scaler = StandardScaler()
                    x_balanced_scaled = scaler.fit_transform(x_balanced)
                    x_test_scaled = scaler.transform(x_test)
                else:
                    x_balanced_scaled = x_balanced
                    x_test_scaled = x_test

                # Initialize a new set of classifiers
                classifiers = Classifiers(random_state=random_state)

                # For each classifier
                for classifier in tqdm(classifiers.models_, desc="Classifying...      "):
                    reset_random_states(np_random_state, torch_random_state, cuda_random_state)

                    classifier.fit(x_balanced_scaled, y_balanced)
                    y_predict = classifier.predict(x_test_scaled)

                    for scorer in scorers:
                        # Binary classification evaluation
                        if dataset.num_classes < 3:
                            performance = scorers[scorer](y_test, y_predict)
                        # MulTi-class classification evaluation
                        else:
                            metric_arguments = inspect.signature(scorers[scorer]).parameters
                            if 'average' in metric_arguments:
                                performance = scorers[scorer](y_test, y_predict, average='micro')
                            else:
                                performance = scorers[scorer](y_test, y_predict)

                        lst = [key, n_fold, synthesizer.name_, classifier.name_, scorer, performance]
                        performance_list.append(lst)

                    lst = [key, n_fold, synthesizer.name_, classifier.name_, "Fit Time", oversampling_duration]
                    performance_list.append(lst)

            d_drh = ResultHandler(description=paths.resampling_filename + key + "_seed_" + str(random_state),
                                  cv_results=performance_list, out_path=paths.resampling_path_split_files)
            d_drh.record_results()


# To evaluate how hard it is to distinguish between real and synthetic instances, we:
# 1. Create a synthetic dataset with the same number of samples and class distribution as the original one.
#    We mark the synthetic samples with flag 0.
# 2. We mark the original samples with flag 1.
# 3. Merge and shuffle the datasets -> create a new dataset.
# 4. Train a classifier on the new dataset and try to predict the flag. The easier it is to predict the flag, the
#    more distinguishable between real and synthetic data.
def eval_detectability(datasets, num_folds=5, transformer=None, random_state=0):
    """
    Evaluate the ability of a generative model to produce high-fidelity data.
    In this experiment we:
     1. Mark the original samples with a tag label=1.
     2. Create a synthetic dataset with the same number of samples and class distribution as the original one.
        Mark the synthetic samples with a tag label=0.
     3. Merge and shuffle the real and synthetic datasets to produce a new dataset.
     4. Train a classifier on the new dataset and try to predict the tag label. The easier it is to predict the flag,
        the more distinguishable between real and synthetic data. Low classification performance reflects highly
        realistic synthetic data.

    Args:
        datasets (dict): The datasets to be used for evaluation.
        num_folds (int): The number of cross validation folds.
        transformer (str): Determines if/how the balanced data will be normalized.
        random_state: Controls random number generation. Set this to a fixed integer to get reproducible results.
    """
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    # Determine the evaluation measures to be used - Fit time is not included here.
    scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }

    n_dataset, n_fold = 0, 0

    # For each dataset
    for key in datasets.keys():
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        n_dataset += 1
        # if num_dataset > 1:
        #     break

        # Load the dataset from the input CSV file
        ds = datasets[key]
        dataset = TabularDataset(key, class_column=ds['class_col'], categorical_columns=ds['categorical_cols'],
                                 random_state=random_state)
        dataset.load_from_csv(path=ds['path'])
        performance_list = []

        print("\n===================================================================================================")
        print("Detectability experiment")
        dataset.display_params()

        # A SingleTableMetadata() object is required by the SDV models
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(dataset.df_)
        k = list(metadata.columns.keys())[len(metadata.columns.keys()) - 1]
        for k in metadata.columns.keys():
            if k in dataset.categorical_columns:
                metadata.columns[k] = {'sdtype': 'categorical'}
            else:
                metadata.columns[k] = {'sdtype': 'numerical'}
        # The last column becomes categorical - This structure is required by the SDV models.
        metadata.columns[k] = {'sdtype': 'categorical'}

        # Find the class distribution of the dataset and store it into a dictionary. Then pass the dictionary
        # as an argument to the sampling_strategy property of the Data Samplers.
        unique, counts = np.unique(dataset.y_, return_counts=True)
        res_dict = dict(zip(unique, 2 * counts))
        synthesizers = TestSynthesizers(metadata, sampling_strategy=res_dict, random_state=random_state)

        # Label the real data with '1'
        real_labels = np.ones(dataset.num_rows)

        # For each sampler
        all_train_idx = [*range(dataset.num_rows)]

        num_synthesizer = 0
        for synthesizer in synthesizers.over_samplers_:
            num_synthesizer += 1
            if synthesizer.name_ != 'None':
                reset_random_states(np_random_state, torch_random_state, cuda_random_state)
                print("\t\tSampler: ", synthesizer.name_)

                t_s = time.time()

                # Generate synthetic data with the sampler
                x_resampled, y_resampled = synthesizer.fit_resample(dataset=dataset, training_set_rows=all_train_idx,
                                                                    sampling_strategy=res_dict)

                # Although we require from the oversampling method to generate an equal number of samples as those
                # included in the original dataset, several of them (e.g. K-Means SMOTE) may return more. So we
                # must create as many fake labels as the number of generated samples.
                num_generated_samples = y_resampled.shape[0] - dataset.num_rows

                # Label the fake data with '0'
                fake_labels = np.zeros(num_generated_samples)
                real_fake_labels = np.concatenate((real_labels, fake_labels), axis=0)

                oversampling_duration = time.time() - t_s

                # Apply k-fold cross validation
                skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
                n_fold = 0

                # For each fold
                if x_resampled.shape[0] < real_fake_labels.shape[0]:
                    real_fake_labels = real_fake_labels[0:x_resampled.shape[0]]

                for train_idx, test_idx in skf.split(x_resampled, real_fake_labels):
                    n_fold += 1

                    x_train = x_resampled[train_idx]
                    y_train = real_fake_labels[train_idx]
                    x_test = x_resampled[test_idx]
                    y_test = real_fake_labels[test_idx]

                    # Normalize data before feeding it to the classifiers
                    if transformer == 'standardizer':
                        scaler = StandardScaler()
                        x_train_scaled = scaler.fit_transform(x_train)
                        x_test_scaled = scaler.transform(x_test)
                    else:
                        x_train_scaled = x_train
                        x_test_scaled = x_test

                    # Initialize a new set of classifiers
                    classifiers = Classifiers(random_state=random_state)

                    # For each classifier
                    for classifier in classifiers.models_:
                        reset_random_states(np_random_state, torch_random_state, cuda_random_state)

                        classifier.fit(x_train_scaled, y_train)
                        y_predict = classifier.predict(x_test_scaled)

                        for scorer in scorers:
                            performance = scorers[scorer](y_test, y_predict)

                            lst = [key, n_fold, synthesizer.name_, classifier.name_, scorer, performance]
                            performance_list.append(lst)

                        lst = [key, n_fold, synthesizer.name_, classifier.name_, "Fit Time", oversampling_duration]
                        performance_list.append(lst)

        d_drh = ResultHandler(description=paths.detectability_filename + key + "_seed_" + str(random_state),
                              cv_results=performance_list, out_path=paths.detectability_path_split_files)

        d_drh.record_results()


def eval_fidelity(datasets, num_folds=5, transformer=None, random_state=0):
    """
    Evaluate the ability of a generative model to produce high-fidelity data.
    In this experiment we:
     1. Test the performance of a classifier in the original dataset.
     2. Create a synthetic dataset with the same number of samples and class distribution as the original one.
        Test the performance of the classifier in the synthetic dataset.
     3. Compare the difference in the classification performance.  Small differences reflect highly realistic data.

    Args:
        datasets (dict): The datasets to be used for evaluation.
        num_folds (int): The number of cross validation folds.
        transformer (str or None): Determines if/how the balanced data will be normalized.
        random_state: Controls random number generation. Set this to a fixed integer to get reproducible results.
    """
    open(paths.fidelity_path_performance + paths.fidelity_filename + "ErrorLog.txt", 'w').close()

    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    # Determine the evaluation measures to be used - Fit time is not included here.
    scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }

    n_dataset, n_fold = 0, 0

    # For each dataset
    for key in datasets.keys():
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        n_dataset += 1
        # if num_dataset > 1:
        #     break

        # Load the dataset from the input CSV file
        ds = datasets[key]

        dataset = TabularDataset(key, class_column=ds['class_col'], categorical_columns=ds['categorical_cols'],
                                 random_state=random_state)
        dataset.load_from_csv(path=ds['path'])
        performance_list = []

        print("\n===================================================================================================")
        print("Classification performance similarity (Fidelity) experiment")
        dataset.display_params()

        #######################################################################################
        # Begin evaluation of the classifiers in the original dataset
        #######################################################################################
        classifiers = Classifiers(random_state=random_state)

        # Apply k-fold cross validation
        skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
        n_fold = 0

        # A SingleTableMetadata() object is required by the SDV models
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(dataset.df_)
        k = list(metadata.columns.keys())[len(metadata.columns.keys()) - 1]
        for k in metadata.columns.keys():
            if k in dataset.categorical_columns:
                metadata.columns[k] = {'sdtype': 'categorical'}
            else:
                metadata.columns[k] = {'sdtype': 'numerical'}
        # The last column becomes categorical - This structure is required by the SDV models.
        metadata.columns[k] = {'sdtype': 'categorical'}

        print("\t\tClassifying in the original dataset...")

        # For each fold
        for train_idx, test_idx in skf.split(dataset.x_, dataset.y_):
            n_fold += 1
            x_train = dataset.x_[train_idx]
            y_train = dataset.y_[train_idx]

            x_test = dataset.x_[test_idx]
            y_test = dataset.y_[test_idx]

            for classifier in classifiers.models_:
                reset_random_states(np_random_state, torch_random_state, cuda_random_state)

                classifier.fit(x_train, y_train)
                y_predict = classifier.predict(x_test)

                for scorer in scorers:
                    # Binary classification evaluation
                    if dataset.num_classes < 3:
                        performance = scorers[scorer](y_test, y_predict)
                    # MulTi-class classification evaluation
                    else:
                        metric_arguments = inspect.signature(scorers[scorer]).parameters
                        if 'average' in metric_arguments:
                            performance = scorers[scorer](y_test, y_predict, average='micro')
                        else:
                            performance = scorers[scorer](y_test, y_predict)

                    lst = [key, n_fold, "None", classifier.name_, scorer, performance]
                    performance_list.append(lst)

        #######################################################################################
        # Begin evaluation of the classifiers in the synthetic dataset
        #######################################################################################

        # Initialize a new set of data samplers
        synthesizers = TestSynthesizers(metadata, sampling_strategy='create-new', random_state=random_state)

        # For each sampler, fit and resample
        num_synthesizer = 0
        for synthesizer in synthesizers.over_samplers_:
            num_synthesizer += 1
            t_s = time.time()

            reset_random_states(np_random_state, torch_random_state, cuda_random_state)
            print("\t\tSynthesizer: ", synthesizer.name_)

            idx = np.array([i for i in range(dataset.num_rows)])

            # Generate synthetic data with the sampler.
            try:
                x_balanced, y_balanced = synthesizer.fit_resample(dataset=dataset, training_set_rows=idx,
                                                                  sampling_strategy='create-new')
            except (ValueError, RuntimeError) as e:
                print("\t\t\t == Exception Caught: '", e, "' ==")
                fh = open(paths.fidelity_path_performance + paths.fidelity_filename + "ErrorLog.txt", "a")
                fh.write(" == Exception Caught: '" + str(e) + "' ==\n")
                fh.write("\tSynthesizer: " + synthesizer.name_ + "\n")
                fh.write("\tDataset: " + dataset.get_name() + "\n")
                fh.write("\tFold: " + str(n_fold) + "\n")
                fh.write("\tSeed: " + str(random_state) + "\n\n")
                fh.close()
                continue

            oversampling_duration = time.time() - t_s

            skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
            n_fold = 0

            class_encoder = LabelEncoder()
            y_balanced = class_encoder.fit_transform(y_balanced)

            for c_train_idx, c_test_idx in skf.split(x_balanced, y_balanced):
                n_fold += 1
                x_c_train = x_balanced[c_train_idx]
                y_c_train = y_balanced[c_train_idx]

                x_c_test = x_balanced[c_test_idx]
                y_c_test = y_balanced[c_test_idx]

                if transformer == 'standardizer':
                    scaler = StandardScaler()
                    x_balanced_scaled = scaler.fit_transform(x_c_train)
                    x_test_scaled = scaler.transform(x_c_test)
                else:
                    x_balanced_scaled = x_c_train
                    x_test_scaled = x_c_test

                # Initialize a new set of classifiers
                bal_classifiers = Classifiers(random_state=random_state)

                # For each classifier
                for classifier in bal_classifiers.models_:
                    reset_random_states(np_random_state, torch_random_state, cuda_random_state)

                    classifier.fit(x_balanced_scaled, y_c_train)
                    y_predict = classifier.predict(x_test_scaled)

                    for scorer in scorers:
                        # Binary classification evaluation
                        if dataset.num_classes < 3:
                            performance = scorers[scorer](y_c_test, y_predict)
                        # MulTi-class classification evaluation
                        else:
                            metric_arguments = inspect.signature(scorers[scorer]).parameters
                            if 'average' in metric_arguments:
                                performance = scorers[scorer](y_c_test, y_predict, average='micro')
                            else:
                                performance = scorers[scorer](y_c_test, y_predict)

                        lst = [key, n_fold, synthesizer.name_, classifier.name_, scorer, performance]
                        performance_list.append(lst)

                    lst = [key, n_fold, synthesizer.name_, classifier.name_, "Fit Time", oversampling_duration]
                    performance_list.append(lst)

            d_drh = ResultHandler(description=paths.fidelity_filename + key + "_seed_" + str(random_state),
                                  cv_results=performance_list, out_path=paths.fidelity_path_split_files)

            d_drh.record_results()


# This function uses an ImbLearn Pipeline. Each Oversampling/Under-sampling method MUST support the fit_resample method
# To use plug-and-play implementations that do not implement fit_resample, please use eval_resampling.
# This method has been used in the experiments of the paper:
# L. Akritidis, P. Bozanis, "A Clustering-Based Resampling Technique with Cluster Structure Analysis for Software Defect
# Detection in Imbalanced Datasets", Information Sciences, vol. 674, pp. 120724, 2024.
def eval_oversampling_efficacy(datasets, num_threads, random_state):
    """Test the ability of a Generator to improve the performance of a classifier by balancing an imbalanced dataset.
    The Generator performs over-sampling on the minority classes and equalizes the number of samples per class.
    This function uses an ImbLearn Pipeline. Each Oversampling/Under-sampling method MUST support the fit_resample
    method. To use plug-and-play implementations that do not implement fit_resample, please use eval_resampling.
    This method has been used in the experiments of the paper:

    * L. Akritidis, P. Bozanis, "A Clustering-Based Resampling Technique with Cluster Structure Analysis for Software
      Defect Detection in Imbalanced Datasets", Information Sciences, vol. 674, pp. 120724, 2024.

    * Algorithm:

      - For each dataset d, for each classifier `c`, for each sampler `s`.
      - Fit `s`.
      - `d_balanced` <--- over-sample(`d` with `s`).
      - Test classification performance of `c` on `d_balanced`.
      - Steps 2-4 are embedded in a Pipeline; the Pipeline is cross validated with 5 folds.
    """
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    classifiers = Classifiers(random_state=random_state)

    results_list = []

    order = 0
    # For each input dataset
    x = 0
    for key in datasets.keys():
        x += 1
        # if x > 1:
        #     break
        print("\n=================================\n Evaluating dataset", key, "\n=================================\n")
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        ds = datasets[key]
        original_dataset = TabularDataset(key, random_state=random_state)
        original_dataset.load_from_csv(path=ds['path'])

        dataset_results_list = []

        # Convert all columns to numerical
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(original_dataset.df_)
        for k in metadata.columns.keys():
            metadata.columns[k] = {'sdtype': 'numerical'}
        metadata.columns[original_dataset.class_column - 1] = {'sdtype': 'categorical'}

        # For each classifier
        for clf in classifiers.models_:
            synthesizers = TestSynthesizers(metadata, sampling_strategy='auto', random_state=random_state)

            # For each over-sampler, balance the input dataset. The fit_resample method of each sampler is called
            # internally by the `imblearn` pipeline and the cross validator.
            for s in synthesizers.over_samplers_:

                reset_random_states(np_random_state, torch_random_state, cuda_random_state)
                order += 1

                print("Testing", clf.name_, "with", s.name_)

                # pipe_line = make_pipeline(s.sampler_, StandardScaler(), clf.model_)
                pipe_line = make_pipeline(s, clf.model_)
                r, _ = original_dataset.cross_val(estimator=pipe_line, num_folds=5, num_threads=num_threads,
                                                  classifier_str=clf.name_, sampler_str=s.name_, order=order)

                for e in range(len(r)):
                    results_list.append(r[e])
                    dataset_results_list.append(r[e])

        # Record the results for this dataset
        drh = ResultHandler(description=key + "_oversampling", cv_results=dataset_results_list,
                            out_path=paths.detectability_path_split_files)
        drh.record_results()

    # Record the results for all datasets
    drh = ResultHandler(description="oversampling_seed_" + str(random_state),
                        cv_results=results_list, out_path=paths.detectability_path_split_files)

    drh.record_results()


# Experiments on discretization
def eval_discretization(datasets, num_folds=5, transformer=None, random_state=0):
    """
    The effects of discretization in classification

    Args:
        datasets (dict): The datasets to be used for evaluation.
        num_folds (int): The number of cross validation folds.
        transformer (str or None): Determines if/how the balanced data will be normalized.
        random_state: Controls random number generation. Set this to a fixed integer to get reproducible results.

    Returns:
    """
    # Truncate the error log
    open(paths.discr_path_performance + "DiscretizationErrorLog.txt", 'w').close()

    # Initialize the random number generators
    set_random_states(random_state)
    np_random_state, torch_random_state, cuda_random_state = get_random_states()

    # Determine the evaluation measures to be used - Fit time is not included here.
    scorers = {
        'accuracy': accuracy_score,
        'balanced_accuracy': balanced_accuracy_score,
        'sensitivity': sensitivity_score,
        'specificity': specificity_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }

    # Determine the discretization methods
    discretization_methods = {
        # 'Method: None': (None, None, None),
        # 'Method: ChiMerge - Bins: Auto - Weights: None': ('chi-merge', 'auto-bgm', None),
        # 'Method: Uniform - Bins: Auto - Weights: None': ('bins-uni', 'auto-bgm', None),
        # 'Method: Quantile - Bins: Auto - Weights: None': ('bins-q', 'auto-bgm', None),
        # 'Method: KMeans - Bins: Auto - Weights: None': ('bins-k', 'auto-bgm', None),
        # 'Method: BGM - Bins: Auto - Weights: None': ('bins-bgm', 'auto-bgm', None),
        # 'Method: ChiMerge - Bins: 5 - Weights: None': ('chi-merge', 5, None),
        # 'Method: Uniform - Bins: 5 - Weights: None': ('bins-uni', 5, None),
        # 'Method: Quantile - Bins: 5 - Weights: None': ('bins-q', 5, None),
        # 'Method: KMeans - Bins: 5 - Weights: None': ('bins-k', 5, None),
        # 'Method: BGM - Bins: 5 - Weights: None': ('bins-bgm', 5, None)
        'Method: CAIM - Bins: None - Weights: None': ('caim', None, None)
    }

    n_dataset, n_fold = 0, 0

    # For each dataset
    for key in datasets.keys():
        reset_random_states(np_random_state, torch_random_state, cuda_random_state)
        n_dataset += 1
        # if num_dataset > 1:
        #     break

        # Load the dataset from the input CSV file
        ds = datasets[key]

        dataset = TabularDataset(key, class_column=ds['class_col'], categorical_columns=ds['categorical_cols'],
                                 random_state=random_state)
        dataset.load_from_csv(path=ds['path'])
        performance_list = []

        print("\n===================================================================================================")
        print("Discretization effectiveness experiment")
        dataset.display_params()

        for d in discretization_methods.keys():
            d_strategy = discretization_methods[d][0]
            d_bins = discretization_methods[d][1]
            d_weights = discretization_methods[d][2]

            print(f"Transforming data with Discretizer: {d_strategy}, Bins: {d_bins}, Weights: {d_weights}")

            # print("Before Discretization:\n", dataset.x_[:20, :])
            if d_strategy is None:
                d_name = "[ None, None, None ]"

                ohe = TabularTransformer(cont_normalizer='None', clip=False)
                ohe.fit(dataset.x_, ds['categorical_cols'])
                discrete_x = ohe.transform(dataset.x_)
            else:
                d_name = "[ " + d_strategy + ", " + str(d_bins) + ", " + str(d_weights) + " ]"
                discretization_method = ctdDiscretizer(strategy=d_strategy, bins=d_bins, bin_weights=d_weights,
                                                       random_state=random_state)
                discretized_x = discretization_method.fit_transform(dataset.x_, dataset.y_, dataset.continuous_columns)

                ohe = TabularTransformer(cont_normalizer='None', clip=False)
                ohe.fit(discretized_x, [c for c in range(dataset.num_columns)])
                discrete_x = ohe.transform(discretized_x)

            # print("After Discretization:\n", discrete_x[:20, :])

            # Apply k-fold cross validation
            skf = StratifiedKFold(n_splits=num_folds, shuffle=False, random_state=None)
            n_fold = 0

            pbar = tqdm(skf.split(discrete_x, dataset.y_), total=skf.get_n_splits(), desc="Classifying")

            # For each fold
            for train_idx, test_idx in pbar:
                # print("\tFold: ", n_fold)
                n_fold += 1
                pbar.set_description("Classifying fold %d" % n_fold)

                x_train = discrete_x[train_idx]
                y_train = dataset.y_[train_idx]

                x_test = discrete_x[test_idx]
                y_test = dataset.y_[test_idx]

                # Normalize data before feeding it to the classifiers
                if transformer == 'standardizer':
                    scaler = StandardScaler()
                    x_balanced_scaled = scaler.fit_transform(x_train)
                    x_test_scaled = scaler.transform(x_test)
                else:
                    x_balanced_scaled = x_train
                    x_test_scaled = x_test

                # Initialize a new set of classifiers
                classifiers = Classifiers(random_state=random_state)

                # For each classifier
                for classifier in classifiers.models_:
                    reset_random_states(np_random_state, torch_random_state, cuda_random_state)

                    classifier.fit(x_balanced_scaled, y_train)
                    y_predict = classifier.predict(x_test_scaled)

                    for scorer in scorers:
                        # Binary classification evaluation
                        if dataset.num_classes < 3:
                            performance = scorers[scorer](y_test, y_predict)

                        # MulTi-class classification evaluation
                        else:
                            metric_arguments = inspect.signature(scorers[scorer]).parameters
                            if 'average' in metric_arguments:
                                performance = scorers[scorer](y_test, y_predict, average='micro')
                            else:
                                performance = scorers[scorer](y_test, y_predict)

                        lst = [key, n_fold, d_name, classifier.name_, scorer, performance]
                        performance_list.append(lst)

                d_drh = ResultHandler(description=paths.discr_filename + key + "_seed_" + str(random_state),
                                      cv_results=performance_list, out_path=paths.discr_path_performance)
                d_drh.record_results()

                pbar.set_description("Completed")
