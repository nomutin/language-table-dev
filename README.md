# [Language-Table [Lynch+ 2023]](https://github.com/google-research/language-table) を使いたい

![python](https://img.shields.io/badge/python-3.10-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## References

- [google-research/language-table](https://github.com/google-research/language-table)
- [Interactive Language: Talking to Robots in Real Time [Lynch+ 2023]](https://interactive-language.github.io/)

## Credits

This project uses code/dataset from [Language-Table](https://github.com/google-research/language-table) which is under the Apache License 2.0.

## インストール

1. [Google Cloud SDK](https://www.faq.idcf.jp/app/answers/detail/a_id/941/c/98) のインストール

    ```sh
    curl https://sdk.cloud.google.com | bash
    gcloud init
    ```

2. [Google Cloud](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_sim/0.0.1%3Btab=objects?prefix=&forceOnObjectsSortingFiltering=false&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))) からプレイデータをダウンロード

    ```sh
    make download
    ```

3. 依存関係のインストール

    ```sh
    make setup
    ```

## `tfrecord` を単一の `np.array` に変換

```sh
python language_table_dev/main.py \
    --min_len 20 \
    --max_len 100 \
    --factor 0.1
```

### パラメータ

- `min_len` この長さ以下のシーケンスはデータセットに含まない.
- `max_len` この長さ以上のシーケンスはデータセットに含まない.全てのシーケンスはこの長さに padding される.
- `factor` 画像(360,640)を何倍して保存するか.
