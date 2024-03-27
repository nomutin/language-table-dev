"""Language-table を `torch.Tensor` として保存する."""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from language_table_dev.process import (
    ActionTransform,
    ObservationTransform,
    action_padding,
    observation_padding,
)
from language_table_dev.tf import episode_generator

if TYPE_CHECKING:
    import numpy as np


path = "gs://gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1"
min_length = 40
max_length = 150


def save_sequence() -> None:
    """`data`ディレクトリ以下にシーケンスごとに行動・観測を保存する."""
    save_dir = Path("data")
    act_transform = ActionTransform()
    obs_transform = ObservationTransform(factor=0.1, pad_size=(0, 14))

    index = 0

    for episode in episode_generator(path=path):
        act_list: list[Tensor] = []
        obs_list: list[Tensor] = []

        for step in episode.as_numpy_iterator():
            if not isinstance(step, dict):
                msg = f"Expected dict, got {type(step)}"
                raise TypeError(msg)

            act_array: np.ndarray = step["action"]
            act_list.append(act_transform(act_array))
            obs_array: np.ndarray = step["observation"]["rgb"]
            obs_list.append(obs_transform(obs_array))

        if len(act_list) > max_length:
            continue

        act = torch.stack(act_list)
        obs = torch.stack(obs_list)
        if act.shape[0] < min_length:
            act = action_padding(act, min_length)
            obs = observation_padding(obs, min_length)

        torch.save(act.clone(), save_dir / f"action_{index}.pt")
        torch.save(obs.clone(), save_dir / f"observation_{index}.pt")
        index += 1


if __name__ == "__main__":
    save_sequence()
