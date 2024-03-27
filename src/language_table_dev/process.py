"""データの前処理・加工."""

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from einops import repeat
from torch import Tensor
from torchvision.transforms import Pad, Resize, ToTensor
from typing_extensions import Self

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ActionTransform:
    """
    行動の`np.ndarray`を正規化し、`torch.Tensor`に変換する.

    NOTE
    ----
    元の関節角度が-0.1~0.1なので、10倍して-1.0~1.0に正規化する.
    """

    def __call__(self: Self, action: np.ndarray) -> torch.Tensor:
        """正規化・テンソルへの変換を行う."""
        action = action * 10.0
        return torch.from_numpy(action).float()


class ObservationTransform:
    """観測の`np.ndarray`をリサイズし、`torch.Tensor`に変換する."""

    def __init__(self: Self, factor: float, pad_size: tuple) -> None:
        self.to_tensor = ToTensor()
        org_height, org_width = 360, 640
        new_size = int(org_height * factor), int(org_width * factor)
        self.minify = Resize(new_size)
        self.pad = Pad(pad_size)

    def __call__(self: Self, observation: np.ndarray) -> Tensor:
        """Resize and Padding and tensoring observation."""
        tensor = self.to_tensor(observation)
        mini_tensor: Tensor = self.minify(tensor)
        padded_tensor: Tensor = self.pad(mini_tensor)
        return padded_tensor


def action_padding(action: Tensor, length: int) -> Tensor:
    """0埋めで長さを`length`にする."""
    padding_length = length - action.shape[0]
    padding_action = torch.zeros(padding_length, action.shape[1])
    return torch.cat([action, padding_action], dim=0)


def observation_padding(observation: Tensor, length: int) -> Tensor:
    """最後の観測を繰り返して長さを`length`にする."""
    padding_length = length - observation.shape[0]
    padding_observation = repeat(
        observation[-1],
        "c h w -> l c h w",
        l=padding_length,
    )
    return torch.cat([observation, padding_observation], dim=0)


def split_train_validation(
    src_dir: Path,
    dst_dir: Path,
    train_ratio: float,
) -> None:
    """学習データと検証データに分割する."""
    dst_dir.mkdir(exist_ok=True)
    train_dir = dst_dir / "train"
    train_dir.mkdir(exist_ok=True)
    val_dir = dst_dir / "validation"
    val_dir.mkdir(exist_ok=True)

    data_size = len(list(src_dir.glob("action_*.pt")))
    train_size = int(data_size * train_ratio)

    for i in tqdm.tqdm(range(data_size)):
        if i < train_size:
            dst_act = train_dir / f"action_{i}.pt"
            dst_obs = train_dir / f"observation_{i}.pt"
        else:
            dst_act = val_dir / f"action_{i - train_size}.pt"
            dst_obs = val_dir / f"observation_{i - train_size}.pt"
        shutil.copy(src_dir / f"action_{i}.pt", dst_act)
        shutil.copy(src_dir / f"observation_{i}.pt", dst_obs)


if __name__ == "__main__":
    src_dir = Path("data")
    dst_dir = Path("language_table_blocktoblock_4block_sim")
    split_train_validation(src_dir, dst_dir, train_ratio=0.9)
