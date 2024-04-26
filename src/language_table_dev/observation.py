"""データの前処理・加工."""

import shutil
from pathlib import Path

import torch
import tqdm


def save_observation(src_dir: Path, dst_dir: Path) -> None:
    data_size = len(list(src_dir.glob("observation_*.pt")))
    dst_dir.mkdir(exist_ok=True)
    count = 0
    for i in tqdm.tqdm(range(data_size)):
        observation_seq = torch.load(src_dir / f"observation_{i}.pt")
        for obs in observation_seq:
            torch.save(obs.clone(), dst_dir / f"observation_{count}.pt")
            count += 1


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

    data_size = len(list(src_dir.glob("observation_*.pt")))
    train_size = int(data_size * train_ratio)

    for i in tqdm.tqdm(range(data_size)):
        if i < train_size:
            dst_obs = train_dir / f"observation_{i}.pt"
        else:
            dst_obs = val_dir / f"observation_{i - train_size}.pt"
        shutil.copy(src_dir / f"observation_{i}.pt", dst_obs)


if __name__ == "__main__":
    src_dir = Path("tmp")
    dst_dir = Path("language_table_blocktoblock_4block_sim_obs")
    # save_observation(src_dir, dst_dir)
    split_train_validation(src_dir, dst_dir, train_ratio=0.9)
