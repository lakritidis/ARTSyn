#base_path = '/media/leo/7CE54B377BB9B18B/'
#out_path = 'dev/Python/'

# base_path = 'C:/Users/Owner/PycharmProjects/'
# out_path = ''

base_path = 'D:/'
out_path = 'dev/Python/'

# Evaluation files output paths
resampling_path_performance = base_path + out_path + 'Experiments/dctdGAN/results/Resampling/'
resampling_path_split_files = resampling_path_performance + 'splits/'
resampling_filename = 'Resampling_'
resampling_path_loss = resampling_path_performance + 'LossFun/'

fidelity_path_performance = base_path + out_path + 'Experiments/dctdGAN/results/Fidelity/'
fidelity_path_split_files = fidelity_path_performance + 'splits/'
fidelity_filename = 'Fidelity_'

detectability_path_performance = base_path + out_path + 'Experiments/dctdGAN/results/Detectability/'
detectability_path_split_files = fidelity_path_performance + 'splits/'
detectability_filename = 'Detectability_'

discr_path_performance = base_path + out_path + 'Experiments/Discretization/results/'
discr_path_split_files = discr_path_performance + 'splits/'
discr_filename = 'caimDiscretization_'

# Dataset input paths
bin_cont = base_path + 'datasets/Imbalanced/bin_continuous/'
bin_disc = base_path + 'datasets/Imbalanced/bin_discrete/'
bin_mix = base_path + 'datasets/Imbalanced/bin_mixed/'

multi_cont = base_path + 'datasets/Imbalanced/multiclass_continuous/'
multi_disc = base_path + 'datasets/Imbalanced/multiclass_discrete/'
multi_mix = base_path + 'datasets/Imbalanced/multiclass_mixed/'
