import itertools
import logging
import pprint
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Set, Tuple, Union, Optional, List, cast, final
import numpy as np
import pandas as pd
from pandas import concat
from sklearn.exceptions import NotFittedError
from loguru import logger
from sklearn.base import BaseEstimator

from core._automl import H2O, AutoML, AutoGluon
from data._domain import Dataset, Task
from data.repository import DatasetRepository, ImbalancedDatasetRepository, OpenMLDatasetRepository
from core._helpers import infer_positive_target_class, train_test_split


class MLBenchmark:
    """
    User interface for performing ML benchmarks.

    This class presents an unified API to run different ML methods on various settings.

    Benchmarking datasets can be utilized by specifying a repository parameter.
    
    Parameters
    ----------
    repository : DatasetRepository, optional
        Name of the dataset repository for a benchmark.
    automl: str, default ag
        Name of the AutoML tool to run a benchmark.
        Supported values: ag (AutoGluon) and h2o.
    metric: str, default f1
        Name of the metric to validate performance of ML models during training. 
        Also used to test performance of the leader model.
        Supported values: f1, f1_macro, f1_weighted, precision, recall, roc_auc, average_precision, balanced_accuracy,
        mcc and accuracy.
    random_state: int, default 42
        Value, used for controlling randomness during model training.
    timeout: int, optional
        Time budget in seconds of AutoML training on a single dataset.
    extra_metrics: list of int, optional
        Names of additional metrics used only to test the leader model on.
    verbosity: int, default 1
        Level of logging verbosity.
    *args
        Additional arguments should be passed as keyword arguments.
    **kwargs
        Arguments specific to the chosen AutoML tool.
    """

    def __init__(
        self,
        repository: Optional[DatasetRepository] = None,
        automl = 'ag',
        metric = 'f1',
        random_state = 42,
        timeout: Optional[int] = None,
        extra_metrics: Optional[List[str]] = None,
        verbosity: int = 1,
        **kwargs
    ):
        self._backend: AutoML
        self._validation_metric: str
        self._seed: int
        self._timeout: Optional[int]
        self._verbosity: int
        # TODO: create a common class for fitted models.
        self._fitted_model = None
        self._test_metrics: Set[str] = set()

        self.verbosity = verbosity
        self.repository = repository
        self.backend = (automl, kwargs)
        self.validation_metric = metric
        self.seed = random_state
        self.timeout = timeout
        self.test_metrics = extra_metrics

        self._configure_logging()

    def _configure_logging(self) -> None:
        if self.verbosity < 2:
            logger.remove()
            logger.add(sys.stdout, level='INFO')

    def run(self) -> None:
        datasets = self.repository.load_datasets()
        
        for dataset in datasets:
            self._run_on_dataset(dataset)

    @logger.catch(reraise=True)
    def _run_on_dataset(self, dataset: Dataset, x_and_y = False) -> None:
        logger.info(f"Run for Dataset(id={dataset.id}, name={dataset.name}).")
        
        if not x_and_y:
            y_label = dataset.x.columns[-1]
            y = dataset.x[y_label]
            x = dataset.x.drop([y_label], axis=1)
        else:
            x = dataset.x
            y = dataset.y
            y_label = y.name
        
        x_train, x_test, y_train, y_test = train_test_split(x, y)
        
        y_train = y_train.astype(object)
        if y_test.dtype == 'category':
            y_test = y_test.cat.codes

        class_belongings = Counter(y_train)
        class_belongings_formatted = '; '.join(f"{k}: {v}" for k, v in class_belongings.items())
        logger.debug(f"Class belongings: {{{class_belongings_formatted}}}")

        pos_class_label = None
        if len(class_belongings) == 2:
            pos_class_label = infer_positive_target_class(class_belongings)

        if x_and_y:
            training_dataset = Dataset(
                id=dataset.id,
                name=dataset.name,
                x=x_train,
                y=y_train,
            )
        else:
            df = pd.concat((x_train, y_train),axis=1)
            training_dataset = Dataset(
                id=dataset.id,
                name=dataset.name,
                x=df
            )

        training_dataset.size = int(x_train.memory_usage(deep=True).sum() / (1024 ** 2))
        logger.debug(f"Train sample size(floored) is {training_dataset.size}mb.")

        validation_metric = self.validation_metric
        if len(class_belongings) > 2 and str(self.backend) == 'AutoGluon':
            validation_metric += '_weighted' 

        task  = Task(
            dataset=training_dataset,
            metric=validation_metric,
            timeout=self.timeout,
            random_state=self.seed
        )

        start_time = time.time()
        self.backend.fit(task)

        time_passed = time.time() - start_time
        logger.info(f"Training took {time_passed // 60} min.")

        y_predicted = self.backend.predict(x_test)

        if str(self.backend) == 'H2O':
            validation_metric += '_weighted'
        logger.debug(f"Test metrics are {self.test_metrics}")
        
        self.backend.score(self.test_metrics, y_test, y_predicted, pos_class_label)

    @property
    def repository(self) -> DatasetRepository:
        return self._repository
    
    @repository.setter
    def repository(self, value: Optional[DatasetRepository]):
        if value is None:
            raise ValueError(
                f"""
                Invalid value of repository parameter:{value}.
                Options available are instances of: [BinaryImbalancedDatasetRepository, OpenMLDatasetRepository].
                """
            )
        self._repository = value
    
    @property
    def validation_metric(self) -> str:
        return self._validation_metric
    
    @validation_metric.setter
    def validation_metric(self, value: str):
        if value not in [
            'f1',
            'f1_macro',
            'f1_weighted',
            'precision',
            'recall',
            'roc_auc',
            'average_precision',
            'balanced_accuracy',
            'mcc',
            'accuracy'
        ]:
            raise ValueError(
                f"""
                Invalid value of metric parameter: {value}.
                Options available: [
                    'f1',
                    'f1_macro',
                    'f1_weighted',
                    'precision',
                    'recall',
                    'roc_auc',
                    'average_precision',
                    'balanced_accuracy',
                    'mcc',
                    'accuracy'].
                """)
        self._validation_metric = value
    
    @property
    def backend(self) -> AutoML:
        return self._backend

    @backend.setter
    def backend(self, value: Tuple[str, Dict[str, Any]]):
        if value[0] == 'ag':
            self._backend = AutoGluon(**value[1])
        elif value[0] == 'h2o':
            self._backend = H2O(**value[1])
        else:
            raise ValueError(
                f"""
                Invalid value of automl parameter: {value[0]}.
                Options available: ['ag', 'h2o'].
                """)
    
    @property
    def seed(self) -> int:
        return self._seed
    
    @seed.setter
    def seed(self, value: int):
        self._seed = value

    @property
    def timeout(self) -> Optional[int]:
        return self._timeout
    
    @timeout.setter
    def timeout(self, value: Optional[int]):
        self._timeout = value

    @property
    def verbosity(self) -> int:
        return self._verbosity
    
    @verbosity.setter
    def verbosity(self, value: int):
        self._verbosity = value

    @property
    def test_metrics(self) -> Set[str]:
        return self._test_metrics

    @test_metrics.setter
    def test_metrics(self, metrics: Optional[List[str]]):
        self.test_metrics.add(self.validation_metric)
        
        if metrics is not None:
            for metric in metrics:
                if metric not in [
                    'f1',
                    'f1_macro',
                    'f1_weighted',
                    'precision',
                    'recall',
                    'roc_auc',
                    'average_precision',
                    'balanced_accuracy',
                    'mcc',
                    'accuracy'
                ]:
                    raise ValueError(
                        f"""
                        Invalid value of the extra_metrics parameter: {metric}.
                        Options available: [
                            'f1',
                            'f1_macro',
                            'f1_weighted',
                            'precision',
                            'recall',
                            'roc_auc',
                            'average_precision',
                            'balanced_accuracy',
                            'mcc',
                            'accuracy'].
                        """)
                self.test_metrics.add(metric)
