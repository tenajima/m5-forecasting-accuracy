import inspect
from dataclasses import dataclass
from typing import Dict, List

import category_encoders as ce
import gokart
import luigi
import numpy as np
import pandas as pd
from tqdm import tqdm

from scripts.dataset.get_dataset import DataSet, GetDatasetOfFold
from scripts.utils import reduce_mem_usage


@dataclass
class DataForTrain:
    train: pd.DataFrame
    test: pd.DataFrame


class FeatureFactory:
    def get_feature_instance(
        self, feature_name: str, fold_num: int
    ) -> gokart.TaskOnKart:

        """特徴量名を指定するとその特徴量クラスのインスタンスを返す
        Args:
            feature_name (str): 特徴量クラスの名前
        Returns:
            gokart.TaskOnKart
        """

        if feature_name in globals():
            return globals()[feature_name](fold_num=fold_num)
        else:
            raise ValueError(f"{feature_name}って特徴量名は定義されてないよ!!!")

    def get_feature_task(
        self, features: List[str], fold_num: int
    ) -> Dict[str, gokart.TaskOnKart]:
        tasks = {}
        for feature in features:
            tasks[feature] = self.get_feature_instance(feature, fold_num)

        return tasks


class GetFoldFeature(gokart.TaskOnKart):
    """ 特徴作成のための基底クラス """

    fold_num = luigi.IntParameter()

    def feature_list(self) -> List[str]:
        """特徴量名リストを取得する"""
        lst: List[str] = []
        for name in globals():
            obj = globals()[name]
            if inspect.isclass(obj) and obj not in [
                tqdm,
                DataForTrain,
                FeatureFactory,
                GetFoldFeature,
                GetFeature,
                Feature,
            ]:
                lst.append(obj.__name__)
        return lst

    def requires(self):
        ff = FeatureFactory()
        features = ["Target", "HisoryAgg"]
        # もしpのfeaturesが空なら全部の特徴量を作る
        if not features:
            features = self.feature_list()
        return ff.get_feature_task(features, fold_num=self.fold_num)

    def output(self):
        return self.make_target(
            f"./feature/feature_fold_{self.fold_num}.pkl", use_unique_id=False
        )

    def run(self):
        data: DataForTrain = self.load("Target")

        for key in self.input().keys():
            print(key)
            if key == "Target":
                continue
            feature: DataForTrain = self.load(key)
            data.train = data.train.join(feature.train)
            data.test = data.test.join(feature.test)

        calendar = pd.read_csv(
            "../input/m5-forecasting-accuracy/calendar.csv", usecols=["d", "date"]
        )
        calendar["date"] = pd.to_datetime(calendar["date"])
        data.train = (
            data.train.reset_index().merge(calendar, on="d").set_index(["id", "d"])
        )
        data.test = (
            data.test.reset_index().merge(calendar, on="d").set_index(["id", "d"])
        )

        self.dump(data)


class GetFeature(luigi.WrapperTask):
    def requires(self):
        return [
            GetFoldFeature(fold_num=1),
            GetFoldFeature(fold_num=2),
            GetFoldFeature(fold_num=3),
            GetFoldFeature(fold_num=4),
        ]


# =================================================================================


class Feature(gokart.TaskOnKart):
    """ 基底クラス """

    index_columns = ["id", "d"]
    predict_column = "target"

    fold_num = luigi.IntParameter()

    def requires(self):
        return {"dataset": GetDatasetOfFold(fold_num=self.fold_num)}


# ==================================================================================


class Target(Feature):
    def run(self):
        dataset: DataSet = self.load("dataset")
        train = dataset.train
        test = dataset.test

        train = reduce_mem_usage(
            train.set_index(self.index_columns)[[self.predict_column]]
        )
        test = reduce_mem_usage(
            test.set_index(self.index_columns)[[self.predict_column]]
        )

        data = DataForTrain(train, test)
        self.dump(data)


class HisoryAgg(Feature):
    """ historyデータのtargetの集計特徴量 """

    def run(self):
        dataset: DataSet = self.load("dataset")

        history = dataset.history.groupby("id").agg(
            {"target": ["sum", "max", "min", "mean", "var", "skew"]}
        )
        history.columns = ["history_" + "_".join(col) for col in history.columns.values]
        history = history.reset_index()

        train = dataset.train[self.index_columns]
        test = dataset.test[self.index_columns]

        train = train.merge(history, on="id", how="left")
        test = test.merge(history, on="id", how="left")

        train = reduce_mem_usage(train.set_index(self.index_columns))
        test = reduce_mem_usage(test.set_index(self.index_columns))

        data = DataForTrain(train, test)
        self.dump(data)
