"""Save language-table as `torch.Tensor`."""

from pathlib import Path

import torch
from tqdm import tqdm

from language_table_dev.tf import episode_generator
from language_table_dev.transform import ActionTransform, ObservationTransform


def save_batch(data_dir: Path) -> None:
    """Save action and observation as `torch.Tensor`."""

    action_dir = data_dir / "action"
    observation_dir = data_dir / "observation"

    for i, episode in enumerate(episode_generator(raw_data_path="data/raw")):
        action_list = []
        observation_list = []

        for step in episode.as_numpy_iterator():
            action = step["action"]
            action = ActionTransform()(action)
            action_list.append(action)
            observation = step["observation"]["rgb"]
            observation = ObservationTransform()(observation)
            observation_list.append(observation)

        action_batch = torch.stack(action_list)
        observation_batch = torch.stack(observation_list)
        torch.save(action_batch, action_dir / f"action_{i}.pt")
        torch.save(observation_batch, observation_dir / f"observation_{i}.pt")


def split_train_val(data_dir: Path) -> None:
    """Split train and validation data."""

    action_dir = data_dir / "action"
    observation_dir = data_dir / "observation"
    train_dir = data_dir / "train"
    val_dir = data_dir / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_length = 6636
    val_length = 1660

    for i in tqdm(range(train_length)):
        action = torch.load(action_dir / f"action_{i}.pt")
        observation = torch.load(observation_dir / f"observation_{i}.pt")
        torch.save(action, train_dir / f"action_{i}.pt")
        torch.save(observation, train_dir / f"observation_{i}.pt")

    for i in tqdm(range(train_length, train_length + val_length)):
        action = torch.load(action_dir / f"action_{i}.pt")
        observation = torch.load(observation_dir / f"observation_{i}.pt")
        torch.save(action, val_dir / f"action_{i - 6636}.pt")
        torch.save(observation, val_dir / f"observation_{i - 6636}.pt")


def main() -> None:
    """Save action and observation as `torch.Tensor`."""
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    save_batch(data_dir=data_dir)
    split_train_val(data_dir=data_dir)


if __name__ == "__main__":
    main()
