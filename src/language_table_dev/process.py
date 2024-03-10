"""データの前処理・加工."""

import os
import shutil
from pathlib import Path

import numpy as np
import torch
import tqdm
from einops import repeat
from torch import Tensor
from torchvision.transforms import Resize, ToTensor
from typing_extensions import Self

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
        observation_tensor = ToTensor()(np.array(observation))
        mini_tensor: Tensor = Resize((36, 64))(observation_tensor)
        return mini_tensor


def pad_length(src_dir: Path, dst_dir: Path, length: int) -> None:
    """長さを`length`にする. `length`以上のデータは消す."""
    dst_dir.mkdir(exist_ok=True)

    act_size = len(list(src_dir.glob("action_*.pt")))
    obs_size = len(list(src_dir.glob("observation_*.pt")))
    assert act_size == obs_size
    count = 0
    for i in tqdm.tqdm(range(act_size)):
        src_act = torch.load(src_dir / f"action_{i}.pt")
        if src_act.shape[0] >= length:
            continue
        src_obs = torch.load(src_dir / f"observation_{i}.pt")
        dst_act = action_padding(src_act, length)
        dst_obs = observation_padding(src_obs, length)
        torch.save(dst_act.clone(), dst_dir / f"action_{count}.pt")
        torch.save(dst_obs.clone(), dst_dir / f"observation_{count}.pt")
        count += 1


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
