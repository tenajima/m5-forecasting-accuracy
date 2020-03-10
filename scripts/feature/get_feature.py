import inspect
from dataclasses import dataclass
from scripts.dataset.read_data import ReadAndTransformData
from typing import Dict, List

import gokart
import pandas as pd
from tqdm import tqdm

from scripts.utils import reduce_mem_usage


@dataclass
class DataForTrain:
    train: pd.DataFrame
    test: pd.DataFrame


class FeatureFactory:
    def get_feature_instance(self, feature_name: str) -> gokart.TaskOnKart:

        """特徴量名を指定するとその特徴量クラスのインスタンスを返す
        Args:
            feature_name (str): 特徴量クラスの名前
        Returns:
            gokart.TaskOnKart
        """

        if feature_name in globals():
            return globals()[feature_name]()
        else:
            raise ValueError(f"{feature_name}って特徴量名は定義されてないよ!!!")

    def get_feature_task(self, features: List[str]) -> Dict[str, gokart.TaskOnKart]:
        tasks = {}
        for feature in features:
            tasks[feature] = self.get_feature_instance(feature)

        return tasks


class GetFeature(gokart.TaskOnKart):
    """ 特徴作成のための基底クラス """

    def feature_list(self) -> List[str]:
        """特徴量名リストを取得する"""
        lst: List[str] = []
        for name in globals():
            obj = globals()[name]
            if inspect.isclass(obj) and obj not in [
                tqdm,
                DataForTrain,
                FeatureFactory,
                GetFeature,
                Feature,
            ]:
                lst.append(obj.__name__)
        return lst

    def requires(self):
        ff = FeatureFactory()
        features = ["Target", "SimpleKernel", "SimpleTime"]
        # もしpのfeaturesが空なら全部の特徴量を作る
        if not features:
            features = self.feature_list()
        return ff.get_feature_task(features)

    def output(self):
        return self.make_target(f"./feature/feature.pkl", use_unique_id=False)

    def run(self):
        data = self.load("Target")

        for key in self.input().keys():
            print(key)
            if key == "Target":
                continue
            feature: DataForTrain = self.load(key)
            data = data.join(feature.train)

        self.dump(data)


# =================================================================================


class Feature(gokart.TaskOnKart):
    """ 基底クラス """

    index_columns = ["id", "date"]
    predict_column = "demand"

    def requires(self):
        return {"data": ReadAndTransformData()}


# ==================================================================================


class Target(Feature):
    # def requires(self):
    #     return {"data": ReadAndTransformData()}

    def run(self):
        data = self.load("data")
        print("loaded")
        data = data[["id", "demand", "date"]]
        print(data.head())
        self.dump(data)


class SimpleKernel(Feature):
    """
    simple feature from kernel(https://www.kaggle.com/ragnar123/very-fst-model)
    """

    # def requires(self):
    #     return {"data": ReadAndTransformData()}

    def run(self):
        data = self.load("data")
        data = data[["id", "demand", "date", "sell_price"]]

        print("lag calc")
        data["lag_t28"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28)
        )
        data["lag_t29"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(29)
        )
        data["lag_t30"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(30)
        )
        data["rolling_mean_t7"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(7).mean()
        )
        data["rolling_std_t7"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(7).std()
        )
        data["rolling_mean_t30"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(30).mean()
        )
        data["rolling_mean_t90"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(90).mean()
        )
        data["rolling_mean_t180"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(180).mean()
        )
        data["rolling_std_t30"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(30).std()
        )
        data["rolling_skew_t30"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(30).skew()
        )
        data["rolling_kurt_t30"] = data.groupby(["id"])["demand"].transform(
            lambda x: x.shift(28).rolling(30).kurt()
        )

        # price features
        print("price calc")
        data["lag_price_t1"] = data.groupby(["id"])["sell_price"].transform(
            lambda x: x.shift(1)
        )
        data["price_change_t1"] = (data["lag_price_t1"] - data["sell_price"]) / (
            data["lag_price_t1"]
        )
        data["rolling_price_max_t365"] = data.groupby(["id"])["sell_price"].transform(
            lambda x: x.shift(1).rolling(365).max()
        )
        data["price_change_t365"] = (
            data["rolling_price_max_t365"] - data["sell_price"]
        ) / (data["rolling_price_max_t365"])
        data["rolling_price_std_t7"] = data.groupby(["id"])["sell_price"].transform(
            lambda x: x.rolling(7).std()
        )
        data["rolling_price_std_t30"] = data.groupby(["id"])["sell_price"].transform(
            lambda x: x.rolling(30).std()
        )
        data.drop(["rolling_price_max_t365", "lag_price_t1"], inplace=True, axis=1)

        print("reducing...")
        data = reduce_mem_usage(data.set_index(self.index_columns))
        self.dump(data)


class SimpleTime(Feature):
    # def requires(self):
    #     return {"data": ReadAndTransformData()}

    def run(self):
        data = self.load("data")
        data = data[["id", "date"]]

        print("date calculating...")
        data["tmp"] = pd.to_datetime(data["date"])
        data["year"] = data["tmp"].dt.year
        data["month"] = data["tmp"].dt.month
        data["week"] = data["tmp"].dt.week
        data["day"] = data["tmp"].dt.day
        data["dayofweek"] = data["tmp"].dt.dayofweek

        data = data.set_index(self.index_columns)
        data = data[["year", "month", "week", "day", "dayofweek"]]

        print("reducing...")
        data = reduce_mem_usage(data)
        print("dumping...")
        self.dump(data)

