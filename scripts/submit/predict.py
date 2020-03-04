import os

import gokart
import lightgbm as lgb
import luigi
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.train.preprocess import Preprocess
from scripts.train.train import TrainStratifiedKFold


class Predict(gokart.TaskOnKart):
    random_state = luigi.IntParameter()

    def requires(self):
        return {"data": Preprocess(), "models": TrainStratifiedKFold()}

    def output(self):
        save_dir = f"./submit/lgb_{self.random_state}"
        return {
            "oof": self.make_target(os.path.join(save_dir, "oof.csv")),
            "submit": self.make_target(os.path.join(save_dir, "submit.csv")),
            "cv": self.make_target(os.path.join(save_dir, "cv_score.txt")),
            "importance": self.make_target(os.path.join(save_dir, "importance.csv")),
        }

    def run(self):
        data = self.load("data")
        models = self.load("models")

        train_X = data.train_X
        train_y = data.train_y
        test_X = data.test_X
        fold = data.fold
        groups = data.groups

        oof = np.zeros_like(train_y, dtype=float)
        sub = np.zeros_like(test_X.index, dtype=float)
        importance = pd.DataFrame()

        for i, (_, val_idx) in enumerate(fold.split(train_X, train_y, groups)):
            model: lgb.Booster = models[i]
            val_X = train_X.iloc[val_idx]
            oof[val_idx] = model.predict(val_X, num_iteration=model.best_iteration)
            sub += model.predict(test_X)

            df_fold_importance = pd.DataFrame()
            df_fold_importance["feature"] = train_X.columns
            df_fold_importance["fold"] = i
            df_fold_importance["importance"] = model.feature_importance(
                importance_type="gain"
            )
            importance = pd.concat([importance, df_fold_importance])

        sub = sub / fold.get_n_splits()
        out_of_fold = pd.DataFrame(index=train_X.index)
        submit = pd.DataFrame(index=test_X.index)
        submit["target"] = sub
        print(submit.head())
        out_of_fold["target"] = oof

        score = roc_auc_score(train_y, oof)

        importance = importance.groupby("feature")[["importance"]].mean()
        importance["importance"] = importance["importance"].astype(int)
        importance = importance.sort_values("importance", ascending=False).reset_index()

        self.dump(submit.reset_index(), "submit")
        self.dump(out_of_fold.reset_index(), "oof")
        self.dump(str(score), "cv")
        self.dump(importance, "importance")
