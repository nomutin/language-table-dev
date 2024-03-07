"""Transforms to make `np.ndarray` into `torch.Tensor`."""

import numpy as np
import torch
from torchvision.transforms import Pad, Resize, ToTensor
from typing_extensions import Self


class ActionTransform:
    """Transforms `np.ndarray` action to `torch.Tensor` actgon."""

    def __call__(self: Self, action: np.ndarray) -> torch.Tensor:
        """Normalize and tensorize action."""
        action = action + 0.1
        action = action * 5.0
        return torch.from_numpy(action).float()


class ObservationTransform:
    """Transforms `np.ndarray` observation to `torch.Tensor` observation."""

    def __call__(self: Self, observation: np.ndarray) -> torch.Tensor:
        """Resize and Padding and tensorize observation."""
        observation_tensor = ToTensor()(observation)
        mini_observation = Resize((36, 64))(observation_tensor)
        return Pad((0, 14))(mini_observation)
