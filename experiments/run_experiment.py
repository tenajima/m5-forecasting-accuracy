import pandas as pd
from nyaggle.experiment import run_experiment
from nyaggle.validation import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import json


def main():
    data = pd.read_pickle("./resources/feature/feature.pkl")
    data = data.reset_index().set_index("id")
    train = data[data["date"] <= "2016-04-24"]
    test = data[(data["date"] > "2016-04-24")]
    train["date"] = pd.to_datetime(train["date"])

    folds = TimeSeriesSplit(
        "date", times=[(("2015-03-28", "2016-03-28"), ("2016-3-28", "2016-04-25"))]
    )
    try:
        model_params = json.load(open("./model_params.json"))
    except FileNotFoundError:
        model_params = {
            "seed": 236,
            "learning_rate": 0.01,
            "n_estimators": 100000,
            "boosting_type": "gbdt",
            "metric": "rmse",
            "bagging_fraction": 0.75,
            "bagging_freq": 10,
            "colsample_bytree": 0.75,
        }
    fit_params = {"eval_metric": "rmse", "early_stopping_rounds": 100, "verbose": 100}

    run_experiment(
        model_params=model_params,
        X_train=train.drop(columns="demand"),
        y=train["demand"],
        X_test=test.drop(columns="demand"),
        cv=folds,
        eval_func=mean_squared_error,
        type_of_target="continuous",
        fit_params=fit_params,
        with_auto_hpo=False,
        with_mlflow=True,
    )


if __name__ == "__main__":
    main()
