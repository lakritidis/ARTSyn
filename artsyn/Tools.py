import numpy as np
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
