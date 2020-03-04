import os

import pandas as pd
from dotenv import load_dotenv
from nyaggle.experiment import run_experiment
from sklearn.metrics import roc_auc_score
import click

@click.command()
@click.option("--tuning", default="false", help="tuning with lightgbm_tuner [true, false]")
@click.option("--type_of_target", default="auto", help="set `binary`, `continuous`, `multiclass`")
def nyaggle(tuning, type_of_target):
    load_dotenv()

    data = pd.read_pickle("./resources/preprocess/preprocessed_data.pkl")
    if tuning == "true":
        with_auto_hpo = True
    elif tuning == "false":
        with_auto_hpo = False
    else:
        raise ValueError("miss tuning type! Only support `true` or `false`!")

    fit_params = {"eval_metric": "auc", "early_stopping_rounds": 100, "verbose": 100}

    model_params = {
        "seed": os.environ["RANDOM_STATE"],
        "learning_rate": 0.01,
        "n_estimators": 100000,
        "verbose_evals": 100,
#===========================================
        "lambda_l1": 0.008326236276901882,
        "lambda_l2": 6.599312336484268,
        "num_leaves": 3,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.9094149008241834,
        "bagging_freq": 5,
        "min_child_samples": 50,
    }

    run_experiment(
        model_params=model_params,
        X_train=data.train_X,
        y=data.train_y,
        X_test=data.test_X,
        eval_func=roc_auc_score,
        type_of_target=type_of_target,
        cv=data.fold,
        fit_params=fit_params,
        with_auto_hpo=with_auto_hpo,
        sample_submission=pd.read_csv(
            "../input/cat-in-the-dat-ii/sample_submission.csv"
        ),
        submission_filename="submission.csv",
        with_mlflow=True,
    )

if __name__ == "__main__":
    nyaggle()
