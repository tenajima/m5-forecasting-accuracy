from datetime import date, timedelta
import pandas as pd
from nyaggle.experiment import run_experiment
from nyaggle.validation import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import json


def main():
    data = pd.read_pickle("./resources/feature/feature.pkl")
    data = data.reset_index().set_index("id")
    train = data[data["date"] < "2016-04-25"]
    test = data[(data["date"] >= "2016-04-25")]
    train["date"] = pd.to_datetime(train["date"])

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

    folds = TimeSeriesSplit("date", times=times)
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
