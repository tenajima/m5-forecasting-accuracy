from dataclasses import dataclass
from typing import Dict, Union

import gokart
import lightgbm as lgb
import luigi
import pandas as pd
from optuna.integration import lightgbm_tuner

from scripts.feature.get_feature import GetFeature


@dataclass
class ModelResult:
    model: lgb.Booster
    params: Dict[str, Union[str, int, float]]
    test: pd.DataFrame


class TrainByStoreId(gokart.TaskOnKart):
    store_id = luigi.Parameter()
    time_budget = luigi.IntParameter()

    def requires(self):
        return {"feature": GetFeature()}

    def output(self):
        return self.make_target(
            f"./store_id/{self.store_id}/result.pkl", use_unique_id=False
        )

    def run(self):
        feature = (
            self.load("feature")
            .query(f"store_id == '{self.store_id}'")
            .reset_index(level=1)
        )

        train = feature.query("date < '2016-03-28'").drop(columns=["store_id", "date"])
        valid = feature.query("'2016-03-28' <= date < '2016-04-25'").drop(
            columns=["store_id", "date"]
        )
        test = feature.query("'2016-04-25' <= date").copy()

        dataset_train = lgb.Dataset(train.drop(columns="demand"), train["demand"])
        dataset_valid = lgb.Dataset(valid.drop(columns="demand"), valid["demand"])

        params = {
            "objective": "regression",
            "seed": 110,
            "learning_rate": 0.01,
            "boosting_type": "gbdt",
            "metric": "rmse",
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "num_leaves": 131,
            "feature_fraction": 0.41600000000000004,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "min_data_in_leaf": 20,
            "min_child_samples": 25,
        }
        print("hoge", self.time_budget)
        if not self.time_budget:
            print("tuningしないよ")
            model = lgb.train(
                params,
                dataset_train,
                num_boost_round=100000,
                valid_sets=[dataset_train, dataset_valid],
                early_stopping_rounds=200,
                verbose_eval=100,
            )
        else:
            print("tuningするよ")
            model = lightgbm_tuner.train(
                params,
                dataset_train,
                num_boost_round=100000,
                valid_sets=[dataset_train, dataset_valid],
                early_stopping_rounds=200,
                verbose_eval=-1,
                time_budget=self.time_budget,
            )

        predict = model.predict(test.drop(columns=["date", "store_id", "demand"]))
        test["demand"] = predict
        test = test.reset_index().set_index(["id", "date"])[["demand"]]

        result = ModelResult(model, params, test)
        self.dump(result)


class TrainAllStoreId(gokart.TaskOnKart):
    stores = [
        "CA_1",
        "CA_2",
        "CA_3",
        "CA_4",
        "TX_1",
        "TX_2",
        "TX_3",
        "WI_1",
        "WI_2",
        "WI_3",
    ]
    time_budget = luigi.IntParameter()

    def requires(self):
        tasks = {
            store_id: TrainByStoreId(store_id=store_id, time_budget=self.time_budget)
            for store_id in self.stores
        }
        print(tasks)
        return tasks

    def output(self):
        return self.make_target("./store_id/result.pkl", use_unique_id=False)

    def run(self):
        result = pd.DataFrame()
        for store_id in self.stores:
            store_result = self.load(store_id).test
            result = pd.concat([result, store_result])

        self.dump(result)
