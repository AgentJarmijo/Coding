from typing import Union

import numpy as np
import torch
from torch import Tensor


def repeat_vector(vector: np.ndarray, repeats: int) -> Tensor:
    """Convert a NumPy vector to a torch tensor and repeat it end-to-end."""
    if not isinstance(vector, np.ndarray):
        raise TypeError("`vector` must be a numpy.ndarray.")
    if repeats < 0:
        raise ValueError("`repeats` must be non-negative.")

    base: Tensor = torch.as_tensor(vector).reshape(-1).clone()

    return base.repeat(repeats) if repeats else base.new_empty((0,), dtype=base.dtype)
