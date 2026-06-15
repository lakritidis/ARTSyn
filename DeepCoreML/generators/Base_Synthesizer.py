import torch


class BaseSynthesizer:
    """`BaseSynthesizer` provides the base class for all generative models.

    Args:
        name: Synthesizer's name
        random_state: An integer for seeding the involved random number generators.
    """
    def __init__(self, name, random_state):
        self._name = name
        self._input_dim = 0                     # Input data dimensionality
        self._n_classes = 0                     # Number of classes in the input dataset
        self._random_state = random_state       # An integer to seed the random number generators

        self._gen_samples_ratio = None          # Array [number of samples to generate per class]
        self._samples_per_class = None          # Array [ [x_train_per_class] ]

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
