"""Compress raw data into `.npy`."""

import argparse

import numpy as np
from language_table_dev.process import (
    expand_actions,
    expand_observations,
    resize_image,
)
from language_table_dev.tf import episode_generator


def main(min_len: int, max_len: int, factor: float) -> None:
    """Load raw data"""
    action_batch_list = []
    observation_batch_list = []

    for episode in episode_generator(raw_data_path="data/raw/"):
        action_list = []
        observation_list = []

        for step in episode.as_numpy_iterator():
            if not isinstance(step, dict):
                msg = f"Expected dict, got {type(step)}"
                raise TypeError(msg)

            action = step["action"]
            observation = step["observation"]["rgb"]
            observation = resize_image(observation, factor=factor)
            action_list.append(action)
            observation_list.append(observation)

        if len(action_list) > max_len or len(action_list) <= min_len:
            continue

        actions = np.stack(action_list, axis=0).astype(np.float16)
        actions = expand_actions(actions=actions, max_len=max_len)
        observations = np.stack(observation_list, axis=0).astype(np.uint8)
        observations = expand_observations(observations, max_len=max_len)

        action_batch_list.append(actions)
        observation_batch_list.append(observations)

    action_batch = np.stack(action_batch_list, axis=0)
    observation_batch = np.stack(observation_batch_list, axis=0)
    action_batch_list.clear()
    observation_batch_list.clear()
    np.save("data/processed/joint_states.npy", action_batch)
    np.save("data/processed/image_states.npy", observation_batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_len", type=int, default=20)
    parser.add_argument("--max_len", type=int, default=100)
    parser.add_argument("--factor", type=float, default=0.1)
    args = parser.parse_args()
    main(
        min_len=args.min_len,
        max_len=args.max_len,
        factor=args.factor,
    )
