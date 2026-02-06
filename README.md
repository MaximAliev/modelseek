# MLB
MLB (stands for machine learning benchmarks) is a library, that allows to conviniently benchmark various machine learning methods on different data modalities. 

It can be utilized by ML engineers and scientists, developing their own method, as well as the common users to test different machine learning scenarios.

## Project status
This project is under active development and new tasks and ML methods to be added soon.

Currently, the only supported task is tabular classification. Also, it currently only supports ML algorithms, that are part of AutoML tools. Specifically, [AutoGluon](https://github.com/autogluon/autogluon) and [H2O](https://github.com/h2oai/h2o-3).

## Installation and usage

### Installation
1. Clone the project.
2. Initialize project with `uv init` and create a virtual environment with `uv venv -p 3.10`.
3. Install dependencies with `uv sync`. For CPU-only installation type `uv sync --extra cpu`. 

### Usage examples

Using a local dataset.
```python
from mlbenchmark.domain import Dataset
from src.mlbenchmark.api import Benchmark
import pandas as pd


path_to_local_data = "datasets/local/ecoli.csv"
dataset = Dataset(name='ecoli', x=pd.read_csv(path_to_local_data))

bench = Benchmark(
    automl='ag',
    metric='f1',
    timeout=60,
    verbosity=2
)
bench.run(dataset)
```

Using a dataset(or collection of such) from a wellknown-source.
```python
from src.mlbenchmark.api import Benchmark
from data.repository import OpenMLDatasetRepository


# WARNING: This OpenML benchmark contains big datasets, that may not fit into your RAM.
datasets = OpenMLDatasetRepository(id=271, verbosity=1).load_datasets(x_and_y=False)
bench = Benchmark(
    automl='ag',
    preset='best',
    metric='f1',
    timeout=360,
    verbosity=1
)

for dataset in datasets:
    bench.run(dataset)
```

## Contribution
Contribution is welcome! Feel free to open issues and submit pull requests.
