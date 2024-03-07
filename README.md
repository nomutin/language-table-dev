# [Language-Table [Lynch+ 2023]](https://github.com/google-research/language-table) as Pytorch Tensor

![python](https://img.shields.io/badge/python-3.10-blue)
[![Rye](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/rye/main/artwork/badge.json)](https://rye-up.com)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## References

- [google-research/language-table](https://github.com/google-research/language-table)
- [Interactive Language: Talking to Robots in Real Time [Lynch+ 2023]](https://interactive-language.github.io/)

## Credits

This project uses code/dataset from [Language-Table](https://github.com/google-research/language-table) which is under the Apache License 2.0.

## Usage

1. Install dependencies ([rye](https://github.com/astral-sh/rye) required)

    ```sh
    make setup
    ```

2. Download the dataset

    See [GCP Paths](https://github.com/google-research/language-table?tab=readme-ov-file#paths).

    ```sh
    make save path=<GCP Path>
    ```
