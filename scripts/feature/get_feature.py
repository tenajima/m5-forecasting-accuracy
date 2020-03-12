import inspect
from dataclasses import dataclass
from scripts.dataset.read_data import ReadAndTransformData
from typing import Dict, List

import gokart
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

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
        features = [
            "Target",
            "SimpleKernel",
            "SimpleTime",
            "SimpleLabelEncode",
        ]
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
            data = data.join(feature)

        self.dump(data)


# =================================================================================


class Feature(gokart.TaskOnKart):
    """ 基底クラス """

    index_columns = ["id", "date"]
    predict_column = "demand"
    to_history_date = "2015-03-27"  # 2015年3月27日までは履歴データとして使う

    def set_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        to_history_date以降のデータに対してindexをsetして、メモリを削減したデータフレームを返す。

        Args:
            data (pd.DataFrame): 特徴量のデータフレーム

        Returns:
            pd.DataFrame:
        """
        data = data.query(f"date > '{self.to_history_date}'")
        data = data.set_index(self.index_columns)
        data = reduce_mem_usage(data)
        return data

    def requires(self):
        return {"data": ReadAndTransformData()}


# ==================================================================================


class Target(Feature):
    def run(self):
        data = self.load("data")
        print("loaded")
        data = data[["id", "demand", "date"]]
        data = self.set_index(data)
        self.dump(data)


class SimpleKernel(Feature):
    """
    simple feature from kernel(https://www.kaggle.com/ragnar123/very-fst-model)
    """

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
        data.drop(
            ["rolling_price_max_t365", "lag_price_t1", "demand"], inplace=True, axis=1
        )

        data = self.set_index(data)
        self.dump(data)


class SimpleTime(Feature):
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

        data = data[["id", "date", "year", "month", "week", "day", "dayofweek"]]

        data = self.set_index(data)
        self.dump(data)


class SimpleLabelEncode(Feature):
    def run(self):
        data = self.load("data")
        cat_columns = [
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]
        use_cols = self.index_columns + cat_columns
        data = data[use_cols]
        nan_features = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        for feature in nan_features:
            data[feature].fillna("unknown", inplace=True)
        for feature in cat_columns:
            encoder = LabelEncoder()
            data[feature] = encoder.fit_transform(data[feature])

        data = self.set_index(data)
        self.dump(data)


class HistoricalDemandAggByItem(Feature):
    groupby_key = ["item_id"]
    """ 履歴のitem_idに関するdemandの統計特徴量 """

    def run(self):
        data = self.load("data")[["id", "date", "demand"] + self.groupby_key]
        history = data.query(f"date <= '{self.to_history_date}'").copy()
        data = data.query(f"date > '{self.to_history_date}'")

        history = history.groupby(self.groupby_key).agg(
            {"demand": ["sum", "max", "std"]}
        )
        columns = [
            "history_" + "_".join(self.groupby_key) + "_" + col[0] + "_" + col[1]
            for col in history.columns.values
        ]
        history.columns = columns

        data = data.merge(history, on=self.groupby_key)

        data = data.drop(columns=["demand"] + self.groupby_key)
        data = self.set_index(data)
        self.dump(data)


class HistoricalDemandAggByDept(HistoricalDemandAggByItem):
    groupby_key = ["dept_id"]


class HistoricalDemandAggByCat(HistoricalDemandAggByItem):
    groupby_key = ["cat_id"]


class HistoricalDemandAggByStore(HistoricalDemandAggByItem):
    groupby_key = ["store_id"]


class HistoricalDemandAggByState(HistoricalDemandAggByItem):
    groupby_key = ["state_id"]


class HistoricalDemandAggByItemStore(HistoricalDemandAggByItem):
    groupby_key = ["item_id", "store_id"]


class HistoricalDemandAggByItemState(HistoricalDemandAggByItem):
    groupby_key = ["item_id", "state_id"]


class HistoricalDemandAggByDeptStore(HistoricalDemandAggByItem):
    groupby_key = ["dept_id", "store_id"]


class HistoricalDemandAggByDeptState(HistoricalDemandAggByItem):
    groupby_key = ["dept_id", "state_id"]


class HistoricalDemandAggByCatStore(HistoricalDemandAggByItem):
    groupby_key = ["cat_id", "store_id"]


class HistoricalDemandAggByCatState(HistoricalDemandAggByItem):
    groupby_key = ["cat_id", "state_id"]


class HistoricalDemandAggByItemMonth(Feature):
    """ 履歴のitem_id, monthに関するdemandの統計特徴量 """

    target_columns = ["item_id"]

    def run(self):
        data = self.load("data")[["id", "date", "demand"] + self.target_columns]
        # yyyy-mm-ddのmmの部分だけ取り出す
        data["month"] = data["date"].str[5:7]
        history = data.query(f"date <= '{self.to_history_date}'").copy()
        data = data.query(f"date > '{self.to_history_date}'")

        groupby_key = ["month"] + self.target_columns
        history = history.groupby(groupby_key).agg({"demand": ["sum", "max", "std"]})
        columns = [
            "history_" + "_".join(groupby_key) + "_" + col[0] + "_" + col[1]
            for col in history.columns.values
        ]
        history.columns = columns

        data = data.merge(history, on=groupby_key)

        data = data.drop(columns=["demand"] + groupby_key)
        data = self.set_index(data)
        self.dump(data)


class HistoricalDemandAggByDeptMonth(HistoricalDemandAggByItemMonth):
    target_columns = ["dept_id"]


class HistoricalDemandAggByCatMonth(HistoricalDemandAggByItemMonth):
    target_columns = ["cat_id"]


class HistoricalDemandAggByStoreMonth(HistoricalDemandAggByItemMonth):
    target_columns = ["store_id"]


class HistoricalDemandAggByStateMonth(HistoricalDemandAggByItemMonth):
    target_columns = ["state_id"]
