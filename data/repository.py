from abc import ABC, abstractmethod
from typing import Optional, Union, List, Any, TypeVar, cast
import numpy as np
import pandas as pd
import openml
from imblearn.datasets import fetch_datasets
from loguru import logger
import itertools

from data.domain import Dataset


class DatasetRepository(ABC):
    def __init__(self, *args, **kwargs):
        self._datasets: List[Dataset] = []

    @abstractmethod
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, X_and_y = False) -> List[Dataset]:
        raise NotImplementedError()

    @abstractmethod
    def load_dataset(self, id: int, X_and_y = False) -> Optional[Dataset]:
        raise NotImplementedError()

    @property
    def datasets(self):
        return self._datasets

class BinaryImbalancedDatasetRepository(DatasetRepository):
    def __init__(self, verbose = False):
        super().__init__()

        # TODO: parallelize.
        self._raw_datasets = fetch_datasets(data_home='datasets/imbalanced', verbose=verbose)

    @logger.catch
    def load_dataset(self, id: int, X_and_y = False) -> Optional[Dataset]:
        for i, (dataset_name, dataset_data) in enumerate(self._raw_datasets.items(), 1):
            if i == id:
                x = dataset_data.get("data")
                y = dataset_data.get("target")[:, np.newaxis]

                if not X_and_y:
                    x = pd.DataFrame(np.concatenate((x, y), axis=1))
                    y = None
                else:
                    # TODO: make this branch work without later fails.
                    x = pd.DataFrame(x)
                    y = pd.Series(y.T[0], dtype=str)
                
                return Dataset(
                    id=id,
                    name=dataset_name,
                    x=x,
                    y=y
                )
            elif i > id:
                raise ValueError(f"Id {id}) is out of range.")

    @logger.catch
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, X_and_y = False) -> List[Dataset]:
        if ids is None:
            range_start = 1
            range_end = len(self._raw_datasets.keys()) + 1
            ids = range(range_start, range_end)
        
        logger.debug(f"Chosen dataset identifiers: {ids}.")
        for id in ids:
            dataset = self.load_dataset(id, X_and_y)
            if dataset is not None:
                self._datasets.append(dataset)
        
        return self.datasets


# REFACTOR THIS SHIT.
class OpenMLDatasetRepository(DatasetRepository):
    def __init__(self, suite_id=271):
        super().__init__()
        self._suite_id = suite_id
        openml.config.set_root_cache_directory("datasets/openml")

    @logger.catch
    def load_dataset(self, id: int, X_and_y = False) -> Optional[Dataset]:
        task = openml.tasks.get_task(id)
        dataset = task.get_dataset()
        x, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute)

        x = cast(np.ndarray, x)
        y = cast(np.ndarray, y)
        if not X_and_y:
            x = pd.DataFrame(np.concatenate((x,y) , axis=1))
            y = None
        else:
            x = pd.DataFrame(x)
            y = pd.Series(y, name=dataset.default_target_attribute, dtype="category")
        
        return Dataset(
            id=id,
            name=dataset.name,
            x=x,
            y=y
        )

    # TODO: parallelize.
    @logger.catch
    def load_datasets(self, ids: Optional[Union[List[int], range]] = None, X_and_y = False) -> List[Dataset]:
        benchmark_suite = openml.study.get_suite(suite_id=self._suite_id)
        if benchmark_suite.tasks is not None:
            for i, id in enumerate(benchmark_suite.tasks):
                if ids is not None and i not in ids:
                    raise ValueError(f"Id {id}) is out of range.")
                
                dataset = self.load_dataset(id, X_and_y)
                if dataset is not None: 
                    self._datasets.append(dataset)
        else:
            raise ValueError("Tasks did not load.")
        
        return self.datasets
