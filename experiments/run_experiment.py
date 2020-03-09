import pandas as pd
from nyaggle.experiment import run_experiment
from nyaggle.validation import Nth
from sklearn.metrics import mean_squared_error
import json


def main():
    data = pd.read_pickle("./resources/feature/feature_fold_3.pkl")
    test_data = pd.read_pickle("./resources/feature/feature_fold_4.pkl")
    train = pd.concat([data.train, data.test])
    folds = Nth(3, pd.read_pickle("./resources/fold/fold.pkl"))
    try:
        model_params = json.load(open("./model_params.json"))
    except FileNotFoundError:
        model_params = {
            "seed": 42,
            "learning_rate": 0.01,
            "n_estimators": 100000,
            "verbose_evals": 100,
        }
    fit_params = {"eval_metric": "rmse", "early_stopping_rounds": 100, "verbose": 100}

    run_experiment(
        model_params=model_params,
        X_train=train.drop(columns="target"),
        y=train["target"],
        X_test=test_data.test.drop(columns="target"),
        cv=folds,
        eval_func=mean_squared_error,
        type_of_target="continuous",
        fit_params=fit_params,
        with_auto_hpo=True,
        with_mlflow=True,
    )


if __name__ == "__main__":
    main()
