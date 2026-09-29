# HF 토론 behavior-1k/2026-challenge-demos #1: [bot] Conversion to Parquet

> 원본: https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/discussions/1
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
- 상태: closed · PR 여부: False · 생성: 2026-07-01T00:11:55.000Z

### [comment] parquet-converter · 2026-07-01T00:11:55.000Z

The parquet-converter bot has created a version of this dataset in the Parquet format in the [`refs/convert/parquet`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/tree/refs%2Fconvert%2Fparquet) branch.

## What is Parquet?

Apache Parquet is a popular columnar storage format known for:

- reduced memory requirement,
- fast data retrieval and filtering,
- efficient storage.

**This is what powers the dataset viewer** on each dataset page and every dataset on the Hub can be accessed with the same code (you can use HF Datasets, ClickHouse, DuckDB, Pandas, PostgreSQL, or Polars, [up to you](https://huggingface.co/docs/dataset-viewer/parquet_process)).

You can learn more about the advantages associated with Parquet in the [documentation](https://huggingface.co/docs/dataset-viewer/parquet).

## How to access the Parquet version of the dataset?

You can access the Parquet version of the dataset by following this link: [`refs/convert/parquet`](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/tree/refs%2Fconvert%2Fparquet)

## What if my dataset was already in Parquet?

When the dataset is already in Parquet format, the data are not converted and the files in `refs/convert/parquet` are links to the original files.

## What should I do?

You don't need to do anything. The Parquet version of the dataset is available for you to use. Refer to the [documentation](https://huggingface.co/docs/dataset-viewer/parquet_process) for examples and code snippets on how to query the Parquet files with ClickHouse, DuckDB, Pandas or Polars.

If you have any questions or concerns, feel free to ask in the discussion below. You can also close the discussion if you don't have any questions.

### [status-change] wensi-ai · 2026-08-05T16:40:35.000Z

{"status": "closed"}
