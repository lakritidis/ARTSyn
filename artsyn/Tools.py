import random
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import torch

import gc
import contextlib


def set_random_states(manual_seed):
    """Initializes the random number generators of NumPy, PyTorch, and PyTorch CUDA by passing the input seed.

    Args:
        manual_seed: An integer to be passed to the random number generators.
    """
    np.random.seed(manual_seed)

    if manual_seed is None:
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
    else:
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_states_2(manual_seed):
    np.random.seed(manual_seed)

    if manual_seed is None:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
    else:
        torch.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
        random.seed(manual_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def get_random_states():
    """Retrieves the current states of randomness of NumPy, PyTorch, and PyTorch CUDA.

    Returns:
        Three states of randomness for NumPy, PyTorch, and PyTorch CUDA respectively.
    """
    np_random_state = np.random.get_state()
    torch_random_state = torch.random.get_rng_state()
    if torch.cuda.is_available():
        cuda_random_state = torch.cuda.random.get_rng_state()
    else:
        cuda_random_state = None

    return np_random_state, torch_random_state, cuda_random_state


def reset_random_states(np_random_state, torch_random_state, cuda_random_state):
    """Sets the current states of randomness of NumPy, PyTorch, and PyTorch CUDA.

    Args:
        np_random_state: The state at which the NumPy random generator will be set.
        torch_random_state: The state at which the PyTorch random generator will be set.
        cuda_random_state: The state at which the PyTorch CUDA random generator will be set.
    """
    np.random.set_state(np_random_state)
    torch.random.set_rng_state(torch_random_state)

    if torch.cuda.is_available():
        torch.cuda.random.set_rng_state(cuda_random_state)
        torch.cuda.empty_cache()

    gc.collect()


@contextlib.contextmanager
def ct_set_random_states(seed, set_model_random_state):
    """Context manager for managing the random state.

    Args:
        seed (int or tuple):
            The random seed or a tuple of (numpy.random.RandomState, torch.Generator).
        set_model_random_state (function):
            Function to set the random state on the model.
    """
    original_np_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    random_np_state, random_torch_state = seed

    np.random.set_state(random_np_state.get_state())
    torch.set_rng_state(random_torch_state.get_state())

    try:
        yield
    finally:
        current_np_state = np.random.RandomState()
        current_np_state.set_state(np.random.get_state())
        current_torch_state = torch.Generator()
        current_torch_state.set_state(torch.get_rng_state())
        set_model_random_state((current_np_state, current_torch_state))

        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)


def random_state(function):
    """Set the random state before calling the function.

    Args:
        function (Callable): The function to wrap around.
    """

    def wrapper(self, *args, **kwargs):
        if self.random_states is None:
            return function(self, *args, **kwargs)

        else:
            with ct_set_random_states(self.random_states, self.set_random_state):
                return function(self, *args, **kwargs)

    return wrapper

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1) + 1e-8)))

def correlation_ratio(categories, measurements):
    categories = pd.Categorical(categories)
    groups = [measurements[categories == cat] for cat in categories.categories]

    grand_mean = np.mean(measurements)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((measurements - grand_mean)**2)

    return np.sqrt(ss_between / (ss_total + 1e-8))

def compute_mixed_matrix(df, cat_cols):
    cols = df.columns
    mat = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)

    num_cols = [c for c in cols if c not in cat_cols]

    for i, col1 in enumerate(cols):
        for j, col2 in enumerate(cols):

            if col1 == col2:
                mat.loc[col1, col2] = 1.0

            elif col1 in num_cols and col2 in num_cols:
                mat.loc[col1, col2] = df[col1].corr(df[col2])

            elif col1 in cat_cols and col2 in cat_cols:
                mat.loc[col1, col2] = cramers_v(df[col1], df[col2])

            else:
                # numeric-categorical
                if col1 in cat_cols:
                    cat, num = col1, col2
                else:
                    cat, num = col2, col1

                mat.loc[col1, col2] = correlation_ratio(df[cat], df[num])

    return mat
