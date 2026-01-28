# variou on different data modalitiess
The project goal is to improve reproducibility of benchmarking of various machine learning methods on different data modalities.

### Project status
The project under active development and new frameworks, tasks, metrics and data repositories to be added soon.
- Supported frameworks: 
    - [AutoGluon](https://github.com/autogluon/autogluon).
    - [H2O-AutoML](https://github.com/h2oai/h2o-3).
- Supported tasks: 
    - Tabular classification.

### Installation and usage

#### Installation
1. Clone the project.
2. Initialize project with `uv init` and create a virtual environment with `uv venv -p 3.10`.
3. Install dependencies with `uv sync`. For CPU-only installation type `uv sync --extra cpu`. 

#### Usage
```python
from core.api import MLBenchmark


def main():
    bench = MLBenchmark(
        automl='ag',
        preset='best',
        metric='f1',
        timeout=1800,
        extra_metrics=['average_precision']
    )
    bench.run()


if __name__ == '__main__':
    main()
```

### Contribution
Contribution is welcome! Feel free to open issues and submit pull requests.
