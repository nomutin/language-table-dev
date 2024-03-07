"""Handle tensorflow datasets."""

from collections.abc import Generator

import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tqdm import tqdm


def episode_generator(raw_data_path: str) -> Generator[DatasetV2, None, None]:
    """
    Generate episodes from raw data stored at a given directory.

    This function creates a TensorFlow datasets (tfds) builder from the
    directory specified by raw_data_path. For each episode in the dataset,
    it checks if the episode is a dict and yields episode steps.

    Parameters
    ----------
    raw_data_path : str
        The path to the directory that contains the raw data.

    Yields
    ------
    Generator[DatasetV2, None, None]
        A generator that yields the steps from each episode in the dataset.

    Raises
    ------
    TypeError
        If the generated dataset is not an instance of DatasetV2.
    TypeError
        If the generated episode is not a dict
    """
    builder = tfds.builder_from_directory(raw_data_path)
    episode_ds = builder.as_dataset(split="train")

    if not isinstance(episode_ds, DatasetV2):
        msg = f"Expected DatasetV2, got {type(episode_ds)}"
        raise TypeError(msg)

    for episode in tqdm(episode_ds):
        if not isinstance(episode, dict):
            msg = f"Expected dict, got {type(episode)}"
            raise TypeError(msg)

        yield episode["steps"]
