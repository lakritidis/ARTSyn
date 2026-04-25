import os
import sys

import numpy as np

import DeepCoreML.eval as eval_methods
import DeepCoreML.paths as paths

num_threads = 1
os.environ['OMP_NUM_THREADS'] = str(num_threads)
np.set_printoptions(linewidth=400, threshold=sys.maxsize)

seed = 42

datasets = {
    # Part 1
    # 'adult': {'path': paths.bin_mix + 'adult.csv', 'categorical_cols': (1, 3, 5, 6, 7, 8, 9, 13), 'class_col': 14},
    #'contraceptive': {'path': paths.multi_mix + 'contraceptive.csv', 'categorical_cols': (4, 5, 8),
    #                  'class_col': 9}, # <-- VERY BAD ACCURACIES FOR ALL
    # 'car-vgood': {'path': paths.bin_disc + 'car-vgood.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5),
    #              'class_col': 6},  # <-- ADASYN FAILS
    # 'CreditCard': {'path': paths.bin_mix + 'creditcarddefault.csv', 'categorical_cols': (1, 2, 3, 5, 6, 7, 8, 9, 10),
    #                 'class_col': 23},
    # 'EyeState': {'path': paths.bin_cont + 'EEG_Eye_State.csv', 'categorical_cols': (), 'class_col': 14},
    # 'flare': {'path': paths.bin_disc + 'flareF.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    #           'class_col': 11},  # <-- COPGAN FAILS
    # 'glass2': {'path': paths.bin_cont + 'glass2.csv', 'categorical_cols': (), 'class_col': 9},
    # 'glass5': {'path': paths.bin_cont + 'glass5.csv', 'categorical_cols': (), 'class_col': 9},
    # 'glass6': {'path': paths.bin_cont + 'glass6.csv', 'categorical_cols': (), 'class_col': 9},
    # 'MobilePrice': {'path': paths.multi_cont + 'MobilePrice.csv', 'categorical_cols': (1, 3, 5, 17, 18, 19),
    #                 'class_col': 20},
    #'obesity': {'path': paths.multi_mix + 'obesity.csv', 'categorical_cols': (0, 4, 5, 6, 8, 9, 11, 14, 15),
    #            'class_col': 16},
    # 'pima': {'path': paths.bin_cont + 'pima.csv', 'categorical_cols': (), 'class_col': 8},
    # 'Thyroid': {'path': paths.multi_cont + 'thyroid.csv', 'categorical_cols': (), 'class_col': 21},  # <-- TVAE FAILS
    # 'vehicle0': {'path': paths.bin_cont + 'vehicle0.csv', 'categorical_cols': (), 'class_col': 18},
    # 'vehicle1': {'path': paths.bin_cont + 'vehicle1.csv', 'categorical_cols': (), 'class_col': 18},
    # 'vehicle2': {'path': paths.bin_cont + 'vehicle2.csv', 'categorical_cols': (), 'class_col': 18},
    # 'vehicle3': {'path': paths.bin_cont + 'vehicle3.csv', 'categorical_cols': (), 'class_col': 18},
    # 'vowel': {'path': paths.bin_mix + 'vowel0.csv', 'categorical_cols': (0, 1, 2), 'class_col': 13},
    # 'yeast6': {'path': paths.bin_cont + 'yeast6.csv', 'categorical_cols': (), 'class_col': 8},
    #'ecoli4': {'path': paths.bin_cont + 'ecoli4.csv', 'categorical_cols': (), 'class_col': 7},

    # Part 2
    'anemia': {'path': paths.multi_cont + 'anemia.csv', 'categorical_cols': (), 'class_col': 14},
    'Churn': {'path': paths.bin_mix + 'Churn_Modelling.csv', 'categorical_cols': (1, 2, 7, 8), 'class_col': 10},
    'DryBean': {'path': paths.multi_cont + 'DryBean.csv', 'categorical_cols': (), 'class_col': 16}, # <-- TVAE FAILS (RESAMPLING)
    'ecoli1': {'path': paths.bin_cont + 'ecoli1.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli2': {'path': paths.bin_cont + 'ecoli2.csv', 'categorical_cols': (), 'class_col': 7},
    'ecoli3': {'path': paths.bin_cont + 'ecoli3.csv', 'categorical_cols': (), 'class_col': 7},
    'FetalHealth': {'path': paths.multi_mix + 'fetal_health.csv', 'categorical_cols': (20,), 'class_col': 21},
    'glass1': {'path': paths.bin_cont + 'glass1.csv', 'categorical_cols': (), 'class_col': 9},
    'glass4': {'path': paths.bin_cont + 'glass4.csv', 'categorical_cols': (), 'class_col': 9},
    'heart': {'path': paths.bin_mix + 'heart.csv', 'categorical_cols': (1, 2, 6, 8, 10, 12), 'class_col': 13},
    'new-thyroid': {'path': paths.multi_cont + 'new-thyroid.csv', 'categorical_cols': (), 'class_col': 5},
    'nursery': {'path': paths.multi_disc + 'nursery.csv', 'categorical_cols': (0, 1, 2, 3, 4, 5, 6, 7), 'class_col': 8},
    'yeast': {'path': paths.multi_cont + 'yeast.csv', 'categorical_cols': (), 'class_col': 8},  # <-- TVAE & ALL SMOTE FAIL
    'yeast1': {'path': paths.bin_cont + 'yeast1.csv', 'categorical_cols': (), 'class_col': 8},
}

datasets_problematic = {
    # Problematic Dat
    # 'hayes-roth': {'path': paths.multi_cont + 'hayes-roth.csv', 'categorical_cols': (), 'class_col': 4},  # <--Errors
    # 'pageblocks': {'path': paths.multi_cont + 'pageblocks.csv', 'categorical_cols': (), 'class_col': 10},  # <--Errors
    # 'penbased': {'path': paths.multi_cont + 'penbased.csv', 'categorical_cols': (), 'class_col': 16},  # <--Errors
    # 'shuttle': {'path': paths.multi_cont + 'shuttle-rev.csv', 'categorical_cols': (), 'class_col': 9},
    # 'autos': {'path': paths.multi_mix + 'autos.csv', 'categorical_cols': (1, 2, 3, 4, 5, 6, 7, 13, 14, 16),
    #          'class_col': 25},  # <--Errors
    # 'balance': {'path': paths.multi_cont + 'balance.csv', 'categorical_cols': (), 'class_col': 4},  # Errors!
    # 'BrainStroke': {'path': paths.bin_mix + 'brain_stroke.csv', 'categorical_cols': (0, 2, 3, 4, 5, 6, 9),
    #                 'class_col': 10}, # <-- VERY LOW F1, COP-GAN FAILS
    # 'LoanModeling': {'path': paths.bin_mix + 'LoanModeling.csv', 'categorical_cols': (5, 7, 8, 9, 10),
    #                  'class_col': 11},  # <-- LOW F1, TVAE FAILS
}

if __name__ == '__main__':
    # eval_methods.test_model('CTDGAN', datasets['ecoli1'], seed)

    eval_methods.eval_resampling(datasets=datasets, transformer='standardizer', num_folds=5, random_state=seed)
    #eval_methods.eval_fidelity(datasets=datasets, transformer=None, num_folds=5, random_state=seed)
    #eval_methods.eval_detectability(datasets=datasets, transformer='standardizer', num_folds=5, random_state=seed)

    # Experiments performed in the Information Sciences 2024 paper
    # eval_methods.eval_oversampling_efficacy(datasets_imb, num_threads, seed)
