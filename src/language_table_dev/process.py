"""データの前処理・加工."""

import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from einops import repeat
from torch import Tensor
from torchvision.transforms import Pad, Resize, ToTensor
from typing_extensions import Self


class ActionTransform:
    """Transforms `np.ndarray` action to `Tensor` action."""

    def __call__(self: Self, action: np.ndarray) -> torch.Tensor:
        """Normalize(clamp -1.0~1.0) and tensoring action."""
        action = action * 10.0
        return torch.from_numpy(action).float()


class ObservationTransform:
    """Transforms `np.ndarray` observation to `Tensor` observation."""

    def __call__(self: Self, observation: np.ndarray) -> torch.Tensor:
        """Resize and Padding and tensoring observation."""
        observation_tensor = ToTensor()(observation)
        mini_observation = Resize((36, 64))(observation_tensor)
        return Pad((0, 14))(mini_observation)


def pad_length(src_dir: Path, dst_dir: Path, length: int) -> None:
    """長さを`length`にする. `length`以上のデータは消す."""
    dst_dir.mkdir(exist_ok=True)

    action_size = len(list(src_dir.glob("action_*.pt")))
    observation_size = len(list(src_dir.glob("observation_*.pt")))
    assert action_size == observation_size
    data_count = 0
    for i in tqdm.tqdm(range(action_size)):
        src_action = src_dir / f"action_{i}.pt"
        src_observation = src_dir / f"observation_{i}.pt"
        action = torch.load(src_action)
        if action.shape[0] >= length:
            continue
        observation = torch.load(src_observation)
        dst_action = action_padding(action, length)
        dst_observation = observation_padding(observation, length)
        torch.save(dst_action, dst_dir / f"action_{data_count}.pt")
        torch.save(dst_observation, dst_dir / f"observation_{data_count}.pt")
        data_count += 1


def action_padding(action: Tensor, length: int) -> Tensor:
    """0埋めで長さを`length`にする."""
    padding_length = length - action.shape[0]
    padding_action = torch.zeros(padding_length, action.shape[1])
    return torch.cat([action, padding_action], dim=0)


def observation_padding(observation: Tensor, length: int) -> Tensor:
    """最後の観測を繰り返して長さを`length`にする."""
    padding_length = length - observation.shape[0]
    last_observation = observation[-1]
    padding_observation = repeat(
        last_observation,
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

    action_size = len(list(src_dir.glob("action_*.pt")))
    observation_size = len(list(src_dir.glob("observation_*.pt")))
    assert action_size == observation_size

    data_size = action_size
    train_size = int(data_size * train_ratio)
    val_size = data_size - train_size
    assert train_size + val_size == action_size

    for i in tqdm.tqdm(range(data_size)):
        src_action = src_dir / f"action_{i}.pt"
        src_observation = src_dir / f"observation_{i}.pt"
        if i < train_size:
            dst_action = train_dir / f"action_{i}.pt"
            dst_observation = train_dir / f"observation_{i}.pt"
        else:
            dst_action = val_dir / f"action_{i - train_size}.pt"
            dst_observation = val_dir / f"observation_{i - train_size}.pt"
        shutil.copy(src_action, dst_action)
        shutil.copy(src_observation, dst_observation)
