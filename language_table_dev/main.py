"""Save language-table as `torch.Tensor`."""

from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Pad, Resize, ToTensor
from typing_extensions import Self

from language_table_dev.tf import episode_generator


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


def main() -> None:
    """Save action and observation as `torch.Tensor`."""
    processed_data_path = Path("data/processed")

    action_dir = processed_data_path / "action"
    observation_dir = processed_data_path / "observation"
    action_dir.mkdir(parents=True, exist_ok=True)
    observation_dir.mkdir(parents=True, exist_ok=True)

    for i, episode in enumerate(episode_generator(raw_data_path="data/raw")):
        action_list = []
        observation_list = []

        for step in episode.as_numpy_iterator():
            action = step["action"]
            observation = step["observation"]["rgb"]
            action_list.append(action)
            observation = ObservationTransform()(observation)
            observation_list.append(observation)

        action_batch = np.stack(action_list)
        observation_batch = np.stack(observation_list)
        torch.save(action_batch, action_dir / f"action_{i}.pt")
        torch.save(observation_batch, observation_dir / f"observation_{i}.pt")


if __name__ == "__main__":
    main()
