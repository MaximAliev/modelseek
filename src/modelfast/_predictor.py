from functools import partial

from catboost import CatBoostClassifier, Pool

from loguru import logger
import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tabpfn import TabPFNClassifier
from tabpfn.finetuning import FinetunedTabPFNClassifier

class Predictor:
    def __init__(self, label, eval_metric, verbosity, seed) -> None:
        self._label = label
        self._eval_metric = eval_metric,
        self._verbosity = verbosity,
        self._seed = seed

    def objective(self, X: pd.DataFrame, y: pd.Series, trial: optuna.Trial):
        # iterations_param = trial.suggest_int("iterations", 30, 3000)
        # learning_rate_param = trial.suggest_float("learning_rate", 1e-5, 1e-1)
        # depth_param = trial.suggest_int("depth", 2, 8, step=2)
        # clf = CatBoostClassifier(
        #     iterations=iterations_param,
        #     learning_rate=learning_rate_param,
        #     depth=depth_param
        # )
        hyperparameters = {"learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4)}
        clf = FinetunedTabPFNClassifier(**hyperparameters)
        trial.set_user_attr("model", clf)

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
        y = LabelEncoder().fit_transform(dataset[self._label])
        X = dataset.drop(labels=[self._label], axis=1)

        default_hyperparameters = {
            "device": "cuda",
        }
        if X.shape[0] < 500:
            default_hyperparameters["balance_probabilities"] = True
        else:
             default_hyperparameters["tuning_config"] = {"calibrate_temperature":True, "tune_decision_thresholds": True}
        
        partial_sampler = optuna.samplers.PartialFixedSampler(default_hyperparameters, study.sampler)
        study.sampler = partial_sampler
        
        study.optimize(partial(self.objective, X, y), n_trials=5)

        logger.success(f"Validation score: {study.best_value}.")
        best_model = study.best_trial.user_attrs.get("model")
        
        best_model.fit(X, y)
        
        return best_model
