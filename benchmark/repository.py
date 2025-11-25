import itertools
import logging
import multiprocessing
import os
import pprint
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, TypeVar

import numpy as np
import pandas as pd
import openml
from imblearn.datasets import fetch_datasets

from core.domain import TabularDataset
from utils.helpers import make_tabular_dataset
from loguru import logger

FittedModel = TypeVar('FittedModel', bound=Any)


class TabularDatasetRepository(ABC):
    def __init__(self, *args, **kwargs):
        self._datasets: List[TabularDataset] = []

    @abstractmethod
    def load_datasets(self, id_range: Optional[List[int]] = None) -> List[TabularDataset]:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, id: Optional[int] = None) -> TabularDataset:
        raise NotImplementedError()

    @property
    def datasets(self):
        return self._datasets

# REFACTOR THIS SHIT.
class ZenodoRepository(TabularDatasetRepository):
    def __init__(self):
        super().__init__()
        self._raw_datasets = fetch_datasets(data_home='datasets/imbalanced', verbose=True)

    @logger.catch
    def load_dataset(self, id: Optional[int] = None) -> TabularDataset:
        for i, (dataset_name, dataset_data) in enumerate(self._raw_datasets.items(), 1):
            if i == id:
                return make_tabular_dataset(
                    name=dataset_name,
                    X=dataset_data.get("data"),
                    y=dataset_data.get("target")
                )
            elif i > id:
                raise ValueError(f"TabularDataset(id={id}) is not available.")

    def load_datasets(self, id_range: Optional[List[int]] = None) -> List[TabularDataset]:
        if id_range is None:
            range_start = 1
            range_end = len(self._raw_datasets.keys()) + 1
            id_range = range(range_start, range_end)
            logger.info(f"Running tasks from {range_start} to {range_end}.")
        for i in id_range:
            self._datasets.append(self.load_dataset(i))
        return self.datasets


# REFACTOR THIS SHIT.
class OpenMLRepository(TabularDatasetRepository):
    def __init__(self, suite_id=271):
        super().__init__()
        self._suite_id = suite_id
        openml.config.set_root_cache_directory("datasets/openml")

    # TODO: parallelize.
    def load_dataset(self, id: Optional[int] = None) -> TabularDataset:
        task = openml.tasks.get_task(id)
        dataset = task.get_dataset()
        X, y, categorical_indicator, feature_names = dataset.get_data(
            target=dataset.default_target_attribute)

        return make_tabular_dataset(
            name=dataset.name,
            y_label=dataset.default_target_attribute,
            X=X,
            y=y
        )

    @logger.catch
    def load_datasets(self, id_range: List[int] = None) -> List[TabularDataset]:
        benchmark_suite = openml.study.get_suite(suite_id=self._suite_id)
        for i, id in enumerate(benchmark_suite.tasks):
            if id_range is not None and i not in id_range:
                raise ValueError(f"TabularDataset(id={id}) is not available.")
            self._datasets.append(self.load_dataset(id))
        return self.datasets
