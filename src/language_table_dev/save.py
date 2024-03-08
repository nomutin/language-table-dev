"""Save language-table as `torch.Tensor`."""

from pathlib import Path
from typing import TYPE_CHECKING

import click
import torch

from language_table_dev.tf import episode_generator
from language_table_dev.transform import ActionTransform, ObservationTransform

if TYPE_CHECKING:
    import numpy as np


@click.command()
@click.option("--path", type=str)
def save_sequence(path: str) -> None:
    """Save action and observation as `torch.Tensor`."""
    save_dir = Path("data")

    for i, episode in enumerate(episode_generator(path=path)):
        act_list = []
        obs_list = []

        for step in episode.as_numpy_iterator():
            if not isinstance(step, dict):
                msg = f"Expected dict, got {type(step)}"
                raise TypeError(msg)

            act_array: np.ndarray = step["action"]
            act_tensor = ActionTransform()(act_array)
            act_list.append(act_tensor)
            obs_array: np.ndarray = step["observation"]["rgb"]
            obs_tensor = ObservationTransform()(obs_array)
            obs_list.append(obs_tensor)

        torch.save(torch.stack(act_list), save_dir / f"action_{i}.pt")
        torch.save(torch.stack(obs_list), save_dir / f"observation_{i}.pt")


if __name__ == "__main__":
    save_sequence()
