from functools import partial

from catboost import CatBoostClassifier, Pool

from loguru import logger
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
# from tabpfn_client import TabPFNClassifier, get_access_token, set_access_token
from tabpfn import TabPFNClassifier

class Predictor:
    def __init__(self, label, eval_metric, verbosity, seed) -> None:
        # self._authenticate()
        self._label = label
        self._eval_metric = eval_metric,
        self._verbosity = verbosity,
        self._seed = seed

    # def _authenticate(self):
        # token = get_access_token()
        # set_access_token(token)

    def objective(self, dataset: pd.DataFrame, trial: optuna.Trial):
        # iterations_param = trial.suggest_int("iterations", 30, 3000)
        # learning_rate_param = trial.suggest_float("learning_rate", 1e-5, 1e-1)
        # depth_param = trial.suggest_int("depth", 2, 8, step=2)
        # clf = CatBoostClassifier(
        #     iterations=iterations_param,
        #     learning_rate=learning_rate_param,
        #     depth=depth_param
        # )
        # n_estimators_param = trial.suggest_int("n_estimators", 8, 80)
        clf = TabPFNClassifier(
            tuning_config={
                "calibrate_temperature": True,
                "tune_decision_thresholds": True,
            },
            # n_estimators=n_estimators_param
        )

        y = LabelEncoder().fit_transform(dataset[self._label])
        X = dataset.drop(labels=[self._label], axis=1)

        val_score = cross_val_score(
            estimator=clf,
            X=X,
            y=y,
            cv=StratifiedKFold(),
            scoring=make_scorer(f1_score, average='binary'),
            error_score='raise'
        ).mean()

        if trial.should_prune():
            logger.info("Trial should prune.")

        return val_score


    def fit(self, dataset):
        study = optuna.create_study(direction='maximize')
        study.optimize(partial(self.objective, dataset), n_trials=3)

        logger.success(study.best_value)
        logger.success(study.best_trial.params)
