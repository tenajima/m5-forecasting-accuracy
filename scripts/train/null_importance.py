import gokart
import luigi
import pandas as pd

from scripts.model.model_lightgbm import train_lgb
from scripts.train.preprocess import DataForML, Preprocess


class _TrainModelForNullImportance(gokart.TaskOnKart):
    model_name = luigi.Parameter()
    random_state = luigi.IntParameter()
    params = luigi.DictParameter()
    shuffle_y = luigi.BoolParameter()

    def requires(self):
        return Preprocess()

    def output(self):
        return self.make_target(
            f"./null_importance/models/{self.model_name}/importance.pkl"
        )

    def run(self):
        data: DataForML = self.load()
        train_X = data.train_X
        train_y = data.train_y
        if self.shuffle_y:
            train_y = train_y.sample(frac=1)
        fold = data.fold

        models = train_lgb(
            fold, self.params.get_wrapped(), train_X, train_y, self.random_state
        )

        importance = pd.DataFrame()
        for i, model in enumerate(models):

            df_fold_importance = pd.DataFrame()
            df_fold_importance["feature"] = train_X.columns
            df_fold_importance["fold"] = i
            df_fold_importance["importance"] = model.feature_importance(
                importance_type="gain"
            )
            importance = pd.concat([importance, df_fold_importance])

        importance = importance.groupby("feature")[["importance"]].mean()
        importance = importance.sort_values("importance", ascending=False).reset_index()
        self.dump(importance)


class NullImportance(gokart.TaskOnKart):
    def requires(self):
        requires_ = dict()
        for i in range(100):
            model_name = "model_" + str(i).zfill(2)
            if i == 0:
                requires_[model_name] = _TrainModelForNullImportance(
                    model_name=model_name, shuffle_y=False
                )
            else:
                requires_[model_name] = _TrainModelForNullImportance(
                    model_name=model_name, shuffle_y=True
                )
        return requires_

    def output(self):
        return self.make_target("./null_importance/importance_cols.txt")

    def run(self):
        importance = pd.DataFrame()

        for i in range(1, 100):
            tmp = self.load(f"model_" + str(i).zfill(2))
            importance = pd.concat([importance, tmp])
        importance["is_normal"] = 0

        normal_importance = self.load("model_00")
        normal_importance["is_normal"] = 1

        importance = pd.concat([normal_importance, importance])

        positive = importance.query("is_normal == 1")
        negative = importance.query("is_normal == 0")

        negative = negative.groupby("feature").agg(
            importance_mean=("importance", "mean"), importance_std=("importance", "std")
        )
        negative["border_line"] = (
            negative["importance_mean"] + negative["importance_std"]
        )

        result = positive.set_index("feature").join(negative)

        result = result.query("importance < border_line").sort_index().index.tolist()
        self.dump(r'"' + '","'.join(result) + r'"')
