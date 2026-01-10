# BAML
The project goal is to improve reproducibility of benchmarking of automated machine learning (AutoML) tools by introducing a unified API.

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
3. Install dependencies with `uv sync`.

#### Usage
```python
from core.runner import BAML


def main():
    baml = BAML(
        automl='ag',
        preset='best',
        metric='f1',
        timeout=1800,
        extra_metrics=['average_precision']
    )
    baml.run()


if __name__ == '__main__':
    main()
```

### Contribution
Contribution is welcome! Feel free to open issues and submit pull requests.

This project is a part of my PhD study and it has no funding at all.
