"""Process observation data."""

import cv2
import numpy as np
from einops import repeat


def resize_image(image: np.ndarray, factor: float) -> np.ndarray:
    """
    Resize a given image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image with shape (height, width, channel).
    factor : float
        The factor to resize the image.

    Returns
    -------
    resized_image : numpy.ndarray
        Resized image with the same shape as the input.
    """
    height, width, _ = image.shape
    new_height = int(height * factor)
    new_width = int(width * factor)
    resized_image: np.ndarray = cv2.resize(
        src=image,
        dsize=(new_width, new_height),
        interpolation=cv2.INTER_LANCZOS4,
    )
    return resized_image


def expand_actions(actions: np.ndarray, max_len: int) -> np.ndarray:
    """
    Expand the actions to a given length.

    Parameters
    ----------
    actions : numpy.ndarray
        The actions to be expanded.
    max_len : int
        The maximum length of the actions.

    Returns
    -------
    expanded_actions : numpy.ndarray
        The expanded actions.
    """
    expand_len = max_len - actions.shape[0]
    last_action = repeat(actions[-1], "c -> t c", t=expand_len)
    return np.concatenate([actions, last_action], axis=0)


def expand_observations(observations: np.ndarray, max_len: int) -> np.ndarray:
    """
    Expand the observations to a given length.

    Parameters
    ----------
    observations : numpy.ndarray
        The observations to be expanded.
    max_len : int
        The maximum length of the observations.

    Returns
    -------
    expanded_observations : numpy.ndarray
        The expanded observations.
    """
    expand_len = max_len - observations.shape[0]
    last_obs = repeat(observations[-1], "h w c -> t h w c", t=expand_len)
    return np.concatenate([observations, last_obs], axis=0)
