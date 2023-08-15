setup:
	poetry install --with core,lint

download:
	cd data/raw && \
	gsutil -m cp \
		"gs://gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1/dataset_info.json" \
		"gs://gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1/features.json" \
		$$(printf "gs://gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1/language_table_blocktoblock_4block_sim-train.tfrecord-%05g " {0..100}) \
		.

lint:
	black . && isort . && ruff --fix .

.PHONY: download lint setup
