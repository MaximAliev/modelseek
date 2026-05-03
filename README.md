# ModelFast

ModelFast is a tool, that automates modelling on your (or benchmark) data, acting as a convinient wrapper to the state-of-the-art automated machine learning (AutoML) libraries. 

It can be utilized by ML engineers, as well as the common users, to test different modelling scenarios.

## Project status
This project is under active development and support for the new tasks, as well as the data modalities, to be added soon.

## Installation and usage

### Installation

1. Clone the project.

2. Initialize project with `uv init` and create a virtual environment with `uv venv -p $version`, where $version >= 3.11.

3. Install dependencies with `uv sync --no-dev`. 

### Usage examples

Using a local dataset.
```python
from src.modelfast.domain import Dataset
from src.modelfast.api import Modeler
import pandas as pd


path_to_local_data = "datasets/local/ecoli.csv"
dataset = Dataset(name='ecoli', x=pd.read_csv(path_to_local_data))

automl = Modeler(
    backend='autogluon',
    metric='f1',
    timeout=3600,
    verbosity=2
)
automl.run(dataset)
```

Using a dataset(or collection of such) from a wellknown-source.
```python
from src.modelfast.api import Modeler
from src.modelfast.repository import OpenMLDatasetRepository


# WARNING: This OpenML benchmark contains big datasets, that may not fit into your RAM.
datasets = OpenMLDatasetRepository(id=271, verbosity=1).load_datasets(x_and_y=False)
automl = Modeler(
    backend='autogluon',
    preset='best',
    metric='f1',
    timeout=3600,
    verbosity=1
)

for dataset in datasets:
    automl.run(dataset)
```

## Contribution
Contribution is welcome! Feel free to open issues and submit pull requests.
