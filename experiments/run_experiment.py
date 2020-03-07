import pandas as pd
from nyaggle.experiment import run_experiment
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import json


def main():
    data = pd.read_pickle("./resources/feature/feature_fold_3.pkl")
    test_data = pd.read_pickle("./resources/feature/feature_fold_4.pkl")
    folds = GroupKFold(n_splits=5)
    groups = data.train.index.get_level_values("id")
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
    print(test_data.test.head())

    run_experiment(
        model_params=model_params,
        X_train=data.train.drop(columns="target"),
        y=data.train["target"],
        X_test=test_data.test.drop(columns="target"),
        cv=folds,
        groups=groups,
        eval_func=mean_squared_error,
        type_of_target="continuous",
        fit_params=fit_params,
        with_auto_hpo=False,
        with_mlflow=True,
    )


if __name__ == "__main__":
    main()
