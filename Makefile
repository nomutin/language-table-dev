help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

clean:  ## 実行に影響のないファイル(.*_cacheとか)を削除
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache/ && \
	rm -f .coverage coverage.xml *.out && \
	find . -type d -name __pycache__ -exec rm -r {} +

format:  ## コードのフォーマット(isort->black->ruff)
	isort . && \
	black . && \
	ruff format .

lint:  ## コードのLint(isort->black->mypy->ruff)
	isort . --check && \
	black . --check && \
	mypy . && \
	ruff check .

setup:  ## 仮想環境の作成
	rye sync --no-lock && \
	rye run pre-commit installj

save:  ## make save path=<GCP Path> でデータの保存
	python src/language_table_dev/save.py --path $(path)

.PHONY: help clean lint format setup
