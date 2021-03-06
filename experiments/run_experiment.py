import copy
import json
import os
import pickle
from datetime import date, datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional

import lightgbm as lgb
import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from optuna.integration import lightgbm_tuner
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator

from nyaggle.validation import TimeSeriesSplit
from scripts.feature import GetFeature
from scripts.evaluate.evaluator import Evaluator


def run_experiment(
    params: Dict,
    X_train: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cv: BaseCrossValidator,
    eval_func: Callable,
    with_auto_hpo: bool = False,
    time_budget: Optional[int] = None,
):

    if with_auto_hpo:
        params = tune_params(params, X_train, y, cv, time_budget=time_budget)

    oof = np.zeros(len(X_train))
    test = np.zeros(len(X_test))

    scores = []
    importance = []
    models = []
    evaluator = Evaluator(load=True)

    for n, (train_idx, valid_idx) in enumerate(cv.split(X_train, y)):
        if "weight" in X_train.columns:
            weight = X_train["weight"].iloc[train_idx]
            del X_train["weight"]
        else:
            weight = None

        dtrain = lgb.Dataset(X_train.iloc[train_idx], y.iloc[train_idx], weight=weight)
        dvalid = lgb.Dataset(X_train.iloc[valid_idx], y.iloc[valid_idx])
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "test"],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=evaluator.feval,
        )

        test += (
            model.predict(X_test, num_iteration=model.best_iteration)
            / cv.get_n_splits()
        )
        oof[valid_idx] = model.predict(
            X_train.iloc[valid_idx], num_iteration=model.best_iteration
        )
        scores.append(evaluator.wrmsse(y.iloc[valid_idx].values, oof[valid_idx]))
        models.append(model)

        importance.append(_get_importance(model, X_train.columns))

    importance = pd.concat(importance)
    importance = (
        importance.groupby("feature")[["importance"]]
        .mean()
        .sort_values("importance", ascending=False)
    )
    test = pd.DataFrame({"demand": test}, index=X_test.index,)

    # 1foldのときだけよ
    if cv.get_n_splits() == 1:
        valid = pd.DataFrame({"y_true": y.iloc[valid_idx], "preds": oof[valid_idx]})
        output_result(models, test, importance, scores, valid)
    else:
        output_result(models, test, importance, scores)


def output_result(
    models: List[lgb.Booster],
    test: pd.DataFrame,
    importance: pd.DataFrame,
    scores: List[float],
    valid=None,
):
    save_dir = os.path.join("./output", datetime.now().strftime(r"%Y%m%d_%H%M%S"))
    os.makedirs(save_dir, exist_ok=True)
    for i, model in enumerate(models):
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        pickle.dump(
            model, open(os.path.join(save_dir, "models", f"model_{i}.pkl"), "wb")
        )
    test.to_csv(os.path.join(save_dir, "test.csv"))

    json.dump(models[0].params, open(os.path.join(save_dir, "params.json"), "w"))

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_title("importance best 20")
    sns.barplot(
        x="importance", y="feature", data=importance.reset_index().head(20), ax=ax
    )
    fig.savefig(os.path.join(save_dir, "importance.png"))

    with open(os.path.join(save_dir, "scores.txt"), "w") as f:
        for i, score in enumerate(scores):
            f.write(f"fold {i}: {score}\n")

    if valid is not None:
        valid.to_csv(os.path.join(save_dir, "oof.csv"))


def tune_params(
    base_param: Dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    time_budget: Optional[int] = None,
) -> Dict:
    train_index, test_index = next(cv.split(X, y))

    dtrain = lgb.Dataset(X.iloc[train_index], y.iloc[train_index])
    dvalid = lgb.Dataset(X.iloc[test_index], y.iloc[test_index])

    params = copy.deepcopy(base_param)
    if "early_stopping_rounds" not in params:
        params["early_stopping_rounds"] = 100

    best_params, tuning_history = dict(), list()
    lightgbm_tuner.train(
        params,
        dtrain,
        valid_sets=[dvalid],
        verbose_eval=0,
        best_params=best_params,
        tuning_history=tuning_history,
        time_budget=time_budget,
    )

    result_param = copy.deepcopy(base_param)
    result_param.update(best_params)
    return result_param


def _get_importance(model: lgb.Booster, features: List[str],) -> pd.DataFrame:
    df = pd.DataFrame()

    df["feature"] = features
    df["importance"] = model.feature_importance(
        importance_type="gain", iteration=model.best_iteration
    )

    return df


def main():
    data = pd.read_pickle("./resources/feature/feature.pkl")
    data = data.reset_index().set_index("id")

    # time_budget = 3600 * 5
    time_budget = 0
    with_auto_hpo = bool(time_budget)

    # 不要な特徴量の排除
    try:
        drop_columns = pd.read_csv("./drop_columns.csv", usecols=["drop_columns"])
        (
            drop_columns.sort_values("drop_columns")
            .reset_index(drop=True)
            .to_csv("./drop_columns.csv", index=False)
        )
        drop_columns = set(drop_columns["drop_columns"])
    except FileNotFoundError:
        drop_columns = set()

    drop_columns = list(set(data.columns) & drop_columns)
    print(drop_columns)
    data = data.drop(columns=drop_columns)

    train = data[data["date"] < "2016-04-25"]
    test = data[(data["date"] >= "2016-04-25")]
    train["date"] = pd.to_datetime(train["date"])

    if "weight" in test.columns:
        test = test.drop(columns="weight")

    valid_type = "short"  # "long", "short", "ts"

    if valid_type == "ts":
        from_test_date = date(2016, 4, 25)
        delta_valid = timedelta(days=28)
        delta_train = timedelta(days=300)
        date_format = r"%Y-%m-%d"
        times = [
            (
                (
                    (from_test_date - delta_valid * (i + 1) - delta_train).strftime(
                        date_format
                    ),
                    (from_test_date - delta_valid * (i + 1)).strftime(date_format),
                ),
                (
                    (from_test_date - delta_valid * (i + 1)).strftime(date_format),
                    (from_test_date - delta_valid * (i)).strftime(date_format),
                ),
            )
            for i in range(5)
        ]
        times = list(reversed(times))
        print(times)
    elif valid_type == "long":
        times = [(("2013-07-18", "2016-03-28"), ("2016-03-28", "2016-04-25"))]
    elif valid_type == "short":
        times = [(("2015-03-28", "2016-03-28"), ("2016-03-28", "2016-04-25"))]
    else:
        raise ValueError("valid_type invalid")

    folds = TimeSeriesSplit(train["date"], times=times)
    try:
        model_params = json.load(open("./model_params.json"))
    except FileNotFoundError:
        model_params = {
            "objective": "rmse",
            "seed": 110,
            "learning_rate": 0.01,
            "n_estimators": 100000,
            "boosting_type": "gbdt",
            "metric": "rmse",
            "bagging_fraction": 0.75,
            "bagging_freq": 10,
            "colsample_bytree": 0.75,
        }

    run_experiment(
        params=model_params,
        X_train=train.drop(columns=["demand", "date"]),
        y=train["demand"],
        X_test=test.drop(columns=["demand", "date"]),
        cv=folds,
        eval_func=lambda y_true, y_pred: mean_squared_error(y_true, y_pred) ** 0.50,
        with_auto_hpo=with_auto_hpo,
    )


if __name__ == "__main__":
    load_dotenv("env")
    if luigi.build([GetFeature()], workers=1, local_scheduler=True):
        main()
