import glob
import inspect
import os
from dataclasses import dataclass
from typing import Dict, List

import gokart
import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from scripts.dataset.read_data import ReadAndTransformData
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
            "SeveralLagFeature",
            "Origin",
            "SimpleTime",
            "SimpleLabelEncode",
            # "SellRatioByStore",
            # "SellRatioByItem",
            "Holiday",
            # "SalesMultipliedSellPriceDevideSellPrice",
            # "ItemSalesPrediction",
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

    index_columns = ["id", "d"]
    predict_column = "sales"
    # to_history_date = "2013-07-17"
    to_history_date = 1518

    def set_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        to_history_date以降のデータに対してindexをsetして、メモリを削減したデータフレームを返す。

        Args:
            data (pd.DataFrame): 特徴量のデータフレーム

        Returns:
            pd.DataFrame:
        """
        data = data.query(f"d > {self.to_history_date}")
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
        data = data[["id", "sales", "d"]]
        data = self.set_index(data)
        self.dump(data)


class Scale(Feature):
    def requires(self):
        return Target()

    def run(self):
        target = self.load().reset_index()[["id", "d"]]
        scale = pd.read_csv("scale.csv")

        result = target.merge(scale, on="id", how="left")
        result = self.set_index(result)
        self.dump(result)


class SimpleKernel(Feature):
    """
    simple feature from kernel(https://www.kaggle.com/ragnar123/very-fst-model)
    """

    def run(self):
        data = self.load("data")
        data = data[["id", "sales", "d", "sell_price"]]

        print("lag calc")
        data["lag_t28"] = data.groupby(["id"])["sales"].transform(lambda x: x.shift(28))
        data["lag_t29"] = data.groupby(["id"])["sales"].transform(lambda x: x.shift(29))
        data["lag_t30"] = data.groupby(["id"])["sales"].transform(lambda x: x.shift(30))
        data["rolling_mean_t7"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(7).mean()
        )
        data["rolling_std_t7"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(7).std()
        )
        data["rolling_mean_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(30).mean()
        )
        data["rolling_std_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(30).std()
        )
        data["rolling_skew_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(30).skew()
        )
        data["rolling_kurt_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(30).kurt()
        )
        data["rolling_mean_t60"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(60).mean()
        )
        data["rolling_mean_t90"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(90).mean()
        )
        data["rolling_std_t90"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(90).std()
        )
        data["rolling_mean_t180"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(180).mean()
        )
        data["rolling_std_t180"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28).rolling(180).std()
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
            ["rolling_price_max_t365", "lag_price_t1", "sales"], inplace=True, axis=1
        )

        # sell_priceのNaN埋め
        data["sell_price"] = data["sell_price"].fillna(-999)
        data = self.set_index(data)
        self.dump(data)


class SellRatioByStore(Feature):
    """
    28日スライドして7日のrollingで得たmeanをもとにstoreごとの合計をとって、そのidがどれだけの割合を占めているかの特徴量
    割り算はlgbが考慮できないので有用だと期待する
    rolling_mean_t7でmeanをとってるけどsumを7で割ったものだし合計と考えて差し支えない
    """

    def requires(self):
        return {"data": SimpleKernel()}

    def run(self):
        data = self.load("data")
        data = data.reset_index()
        data["store_id"] = data["id"].map(lambda x: "_".join(x.split("_")[3:5]))
        data = data[["id", "d", "store_id", "rolling_mean_t7", "rolling_mean_t30"]]

        data = data.merge(
            data.groupby(["d", "store_id"], as_index=False)["rolling_mean_t7"]
            .sum()
            .rename(columns={"rolling_mean_t7": "store_sum_t7"}),
            on=["d", "store_id"],
        )
        data = data.merge(
            data.groupby(["d", "store_id"], as_index=False)["rolling_mean_t30"]
            .sum()
            .rename(columns={"rolling_mean_t30": "store_sum_t30"}),
            on=["d", "store_id"],
        )

        data["ratio_by_store_t7"] = data["rolling_mean_t7"] / data["store_sum_t7"]
        data["ratio_by_store_t30"] = data["rolling_mean_t30"] / data["store_sum_t30"]

        data = data[["id", "d", "ratio_by_store_t7", "ratio_by_store_t30"]]
        data = self.set_index(data)
        self.dump(data)


class SellRatioByItem(Feature):
    """
    28日スライドして7日のrollingで得たmeanをもとにitemごとの合計をとって、そのidがどれだけの割合を占めているかの特徴量
    割り算はlgbが考慮できないので有用だと期待する
    rolling_mean_t7でmeanをとってるけどsumを7で割ったものだし合計と考えて差し支えない
    """

    def requires(self):
        return {"data": SimpleKernel()}

    def run(self):
        data = self.load("data")
        data = data.reset_index()
        data["item_id"] = data["id"].map(lambda x: "_".join(x.split("_")[:3]))
        data = data[["id", "d", "item_id", "rolling_mean_t7", "rolling_mean_t30"]]

        data = data.merge(
            data.groupby(["d", "item_id"], as_index=False)["rolling_mean_t7"]
            .sum()
            .rename(columns={"rolling_mean_t7": "item_sum_t7"}),
            on=["d", "item_id"],
        )
        data = data.merge(
            data.groupby(["d", "item_id"], as_index=False)["rolling_mean_t30"]
            .sum()
            .rename(columns={"rolling_mean_t30": "item_sum_t30"}),
            on=["d", "item_id"],
        )

        data["ratio_by_item_t7"] = data["rolling_mean_t7"] / data["item_sum_t7"]
        data["ratio_by_item_t30"] = data["rolling_mean_t30"] / data["item_sum_t30"]

        data = data[["id", "d", "ratio_by_item_t7", "ratio_by_item_t30"]]
        data = self.set_index(data)
        self.dump(data)


class _SalesLag28(Feature):
    """
    28日前のsalesの特徴量
    """

    def run(self):
        data = self.load("data")
        data = data[["id", "sales", "d"]]

        data["sales_lag"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(28)
        )
        data = data[["id", "d", "sales_lag"]]
        data = self.set_index(data)
        self.dump(data)


class _SellPriceLag28(Feature):
    def run(self):
        data = self.load("data")
        data = data[["id", "sell_price", "d"]]

        data["price_lag"] = data.groupby(["id"])["sell_price"].transform(
            lambda x: x.shift(28)
        )
        data = data[["id", "d", "sell_price", "price_lag"]]
        data = self.set_index(data)
        self.dump(data)


class SalesMultipliedSellPriceDevideSellPrice(gokart.TaskOnKart):
    def requires(self):
        return {"sales": _SalesLag28(), "price": _SellPriceLag28()}

    def run(self):
        data = self.load("sales")
        price = self.load("price")

        data = data.join(price)
        del price

        data["lag_sales_mul_lag_price_dev_price"] = (
            data["sales_lag"] * data["price_lag"] / data["sell_price"]
        )
        data = data[["lag_sales_mul_lag_price_dev_price"]].fillna(-999)
        self.dump(data)


class ItemSalesPrediction(Feature):
    """
    日毎のitemの売上の予測
    """

    def run(self):
        data = self.load("data")
        data = data[["id", "d", "item_id"]]

        predict = pd.read_pickle("./item_result_poisson.pkl")
        data = data.merge(
            predict.rename(columns={"sales": "item_predict"}),
            on=["item_id", "d"],
            how="left",
        )
        data = self.set_index(data[["id", "d", "item_predict"]])
        self.dump(data)


class SeveralLagFeature(Feature):
    """
    simple feature from kernel(https://www.kaggle.com/ragnar123/very-fst-model)
    """

    def run(self):
        data = self.load("data")
        data = data[["id", "sales", "d"]]

        def shift_lag_feature(shift: int):
            data[f"shift_{shift}_rolling_mean_t7"] = data.groupby(["id"])[
                "sales"
            ].transform(lambda x: x.shift(shift).rolling(7).mean())
            data[f"shift_{shift}_rolling_mean_t28"] = data.groupby(["id"])[
                "sales"
            ].transform(lambda x: x.shift(shift).rolling(28).mean())
            data[f"shift_{shift}_rolling_mean_t56"] = data.groupby(["id"])[
                "sales"
            ].transform(lambda x: x.shift(shift).rolling(56).mean())

        tmp_file = glob.glob("./tmp_feature_*.pkl")
        if tmp_file:
            file_name = tmp_file[0]
            restart = int(file_name.split("_")[-1].split(".")[0]) + 1
            print("reload ", file_name)
            data = pd.read_pickle(file_name)
        else:
            restart = 1

        for i in range(restart, 28):
            print("day ", i)
            shift_lag_feature(i)
            data.to_pickle(f"./tmp_feature_{i}.pkl")
            if i != 1:
                os.remove(f"./tmp_feature_{i - 1}.pkl")

        data = data.drop(columns="sales")
        data = self.set_index(data)
        self.dump(data)


class Origin(Feature):
    """ datasetを読み込んで特に手を加えないやつ """

    def run(self):
        data = self.load("data")
        data = data[["id", "d", "snap_CA", "snap_TX", "snap_WI", "wm_yr_wk"]]
        data = self.set_index(data)
        self.dump(data)


class StoreId(Feature):
    """
    store_idごとにモデルを作る際に参考にするぜ
    ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1', 'WI_2', 'WI_3']
    """

    def run(self):
        data = self.load("data")
        data = data[["id", "d", "store_id"]]
        encoder = LabelEncoder()
        data["store_id"] = encoder.fit_transform(data["store_id"])
        data = self.set_index(data)
        self.dump(data)


class SimpleTime(Feature):
    def run(self):
        data = self.load("data")
        data = data[["id", "d", "date"]]

        print("date calculating...")
        data["tmp"] = pd.to_datetime(data["date"])
        data["year"] = data["tmp"].dt.year
        data["month"] = data["tmp"].dt.month
        data["week"] = data["tmp"].dt.week
        data["day"] = data["tmp"].dt.day
        data["dayofweek"] = data["tmp"].dt.dayofweek

        data = data[["id", "d", "month", "week", "day", "dayofweek"]]

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
        data = self.load("data")[["id", "d", "sales"] + self.groupby_key]
        history = data.query(f"date <= '{self.to_history_date}'").copy()
        data = data.query(f"date > '{self.to_history_date}'")

        history = history.groupby(self.groupby_key).agg(
            {"sales": ["sum", "max", "std"]}
        )
        columns = [
            "history_" + "_".join(self.groupby_key) + "_" + col[0] + "_" + col[1]
            for col in history.columns.values
        ]
        history.columns = columns

        data = data.merge(history, on=self.groupby_key)

        data = data.drop(columns=["sales"] + self.groupby_key)
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
        data = self.load("data")[["id", "d", "sales"] + self.target_columns]
        # yyyy-mm-ddのmmの部分だけ取り出す
        data["month"] = data["d"].str[5:7]
        history = data.query(f"date <= '{self.to_history_date}'").copy()
        data = data.query(f"date > '{self.to_history_date}'")

        groupby_key = ["month"] + self.target_columns
        history = history.groupby(groupby_key).agg({"sales": ["sum", "max", "std"]})
        columns = [
            "history_" + "_".join(groupby_key) + "_" + col[0] + "_" + col[1]
            for col in history.columns.values
        ]
        history.columns = columns

        data = data.merge(history, on=groupby_key)

        data = data.drop(columns=["sales"] + groupby_key)
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


class Holiday(Feature):
    def run(self):
        calendar = pd.read_csv(
            "../input/m5-forecasting-accuracy/calendar.csv", usecols=["d", "date"]
        )
        calendar["d"] = calendar["d"].map(lambda x: int(x.split("_")[1]))

        # 前の日、休日が休日か見るために頭とお尻にくっつける
        calendar.loc[calendar.shape[0], "date"] = "2011-01-28"
        calendar.loc[calendar.shape[0], "date"] = "2016-06-20"
        calendar = calendar.sort_values("date").reset_index(drop=True)
        calendar["datetime"] = pd.to_datetime(calendar["date"])

        calendar["dayofweek"] = calendar["datetime"].dt.dayofweek
        calendar["is_weekend"] = calendar["dayofweek"].isin([5, 6]).astype(int)

        us_holidays = holidays.UnitedStates()
        calendar["is_US_holiday"] = (
            calendar["datetime"].map(lambda dt: dt in us_holidays).astype(int)
        )

        calendar["is_day_off"] = calendar["is_weekend"] | calendar["is_US_holiday"]

        calendar["tmp"] = calendar["is_day_off"].diff(periods=-1)
        calendar["before_day_off"] = (calendar["tmp"] == -1).astype(int)
        del calendar["tmp"]

        calendar["tmp"] = calendar["is_day_off"].diff()
        calendar["after_day_off"] = (calendar["tmp"] == -1).astype(int)
        del calendar["tmp"]
        del calendar["datetime"], calendar["dayofweek"]

        data = self.load("data")[["id", "d"]]
        data = data.merge(calendar, on="d")
        data = data[
            [
                "id",
                "d",
                "is_weekend",
                "is_US_holiday",
                "before_day_off",
                "after_day_off",
            ]
        ]
        data = self.set_index(data)
        self.dump(data)


# TODO: 実装したけど、変なことしてないはずなのにスコアが悪化するのでそこを調査する
class AggItemIdMean(Feature):
    groupby_key = ["item_id"]

    def run(self):
        data = self.load("data")[["id", "d", "sales"] + self.groupby_key]

        group = data.groupby(["d"] + self.groupby_key)[["sales"]].mean()

        group["agg_mean_" + "_".join(self.groupby_key) + "_lag_t28"] = group.groupby(
            self.groupby_key
        )["sales"].transform(lambda x: x.shift(28))

        group["agg_mean_" + "_".join(self.groupby_key) + "_lag_t29"] = group.groupby(
            self.groupby_key
        )["sales"].transform(lambda x: x.shift(29))

        group["agg_mean_" + "_".join(self.groupby_key) + "_lag_t30"] = group.groupby(
            self.groupby_key
        )["sales"].transform(lambda x: x.shift(30))

        group[
            "agg_mean_" + "_".join(self.groupby_key) + "_rolling_mean_t7"
        ] = group.groupby(self.groupby_key)["sales"].transform(
            lambda x: x.shift(28).rolling(7).mean()
        )

        group[
            "agg_mean_" + "_".join(self.groupby_key) + "_rolling_std_t7"
        ] = group.groupby(self.groupby_key)["sales"].transform(
            lambda x: x.shift(28).rolling(7).std()
        )

        group[
            "agg_mean_" + "_".join(self.groupby_key) + "_rolling_mean_t30"
        ] = group.groupby(self.groupby_key)["sales"].transform(
            lambda x: x.shift(28).rolling(30).mean()
        )

        group[
            "agg_mean_" + "_".join(self.groupby_key) + "_rolling_std_t30"
        ] = group.groupby(self.groupby_key)["sales"].transform(
            lambda x: x.shift(28).rolling(30).std()
        )

        group[
            "agg_mean_" + "_".join(self.groupby_key) + "_rolling_mean_t90"
        ] = group.groupby(self.groupby_key)["sales"].transform(
            lambda x: x.shift(28).rolling(90).mean()
        )

        group[
            "agg_mean_" + "_".join(self.groupby_key) + "_rolling_mean_t180"
        ] = group.groupby(self.groupby_key)["sales"].transform(
            lambda x: x.shift(28).rolling(180).mean()
        )

        group = group.drop(columns="sales")

        data = data.merge(group, on=["d"] + self.groupby_key)
        data = data.drop(columns=["sales"] + self.groupby_key)
        data = self.set_index(data)
        self.dump(data)


class AggDeptIdMean(AggItemIdMean):
    groupby_key = ["dept_id"]


class AggCatIdMean(AggItemIdMean):
    groupby_key = ["cat_id"]


class AggStoreIdMean(AggItemIdMean):
    groupby_key = ["store_id"]


class AggStateIdMean(AggItemIdMean):
    groupby_key = ["state_id"]


class DaysDiff(Feature):
    def run(self):
        calendar = pd.read_csv(
            "../input/m5-forecasting-accuracy/calendar.csv", usecols=["d"]
        )
        calendar["datetime"] = pd.to_datetime(calendar["d"])
        calendar["days_ago"] = (
            pd.to_datetime("2016-03-28") - calendar["datetime"]
        ).dt.days
        calendar["days_ago"] = calendar["days_ago"].clip(0)

        calendar["days_ago_exp_100"] = np.exp(
            -(np.log(10) / 100) * calendar["days_ago"]
        )
        calendar["days_ago_exp_200"] = np.exp(
            -(np.log(10) / 200) * calendar["days_ago"]
        )
        calendar["days_ago_exp_300"] = np.exp(
            -(np.log(10) / 300) * calendar["days_ago"]
        )
        calendar["days_ago_exp_500"] = np.exp(
            -(np.log(10) / 500) * calendar["days_ago"]
        )
        calendar = calendar.drop(columns="datetime")

        data = self.load("data")[["id", "d"]]
        data = data.merge(calendar, on="d")
        print(data.head())
        data = self.set_index(data)

        self.dump(data)


class WeightRollingMean(Feature):
    def requires(self):
        return {"mean": SimpleKernel(), "weight": DaysDiff()}

    def run(self):
        mean = self.load_data_frame(
            "mean",
            required_columns={
                "rolling_mean_t7",
                "rolling_mean_t30",
                "rolling_mean_t90",
                "rolling_mean_t180",
            },
            drop_columns=True,
        )
        weight = self.load_data_frame(
            "weight",
            required_columns={"days_ago_exp_100", "days_ago_exp_500"},
            drop_columns=True,
        )

        mean = mean.join(weight)

        for mean_term in [7, 30, 90, 180]:
            for weight_param in [100, 500]:
                mean[f"rolling_mean_t{mean_term}_exp{weight_param}"] = (
                    mean[f"rolling_mean_t{mean_term}"]
                    * mean[f"days_ago_exp_{weight_param}"]
                )

        mean = mean.drop(
            columns=[
                "rolling_mean_t7",
                "rolling_mean_t30",
                "rolling_mean_t90",
                "rolling_mean_t180",
                "days_ago_exp_100",
                "days_ago_exp_500",
            ]
        )
        mean = self.dump(mean)


class LongRollingMean(Feature):
    def run(self):
        data = self.load("data")
        data = data[["id", "sales", "d"]]

        data["rolling365_mean_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(365 - 15).rolling(30).mean()
        )
        data["rolling730_mean_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(730 - 15).rolling(30).mean()
        )
        data["rolling1095_mean_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(1095 - 15).rolling(30).mean()
        )
        data["rolling1460_mean_t30"] = data.groupby(["id"])["sales"].transform(
            lambda x: x.shift(1460 - 15).rolling(30).mean()
        )
        data = data.drop(columns="sales")
        data = self.set_index(data)
        self.dump(data)


class GlobalTrend(Feature):
    """ ウォルマート全体の傾向 """

    def run(self):
        data = self.load("data")
        data = data[["id", "sales", "d"]]
        trend = data.groupby("d")[["sales"]].sum().copy()
        trend = (
            trend.reset_index()
            .reset_index()
            .rename(columns={"index": "day"})
            .set_index("d")
        )

        train = trend[trend.index < "2016-03-28"]
        p = np.poly1d(np.polyfit(train["day"], train["sales"], 3))
        trend["global_trend"] = p(trend["day"])
        trend = trend.reset_index()
        data = data.merge(trend[["d", "global_trend"]], on="d", how="left")

        data = self.set_index(data[["id", "d", "global_trend"]])
        self.dump(data)


class ShortLag(Feature):
    """
    28日シフトしないlag
    まずは7日のみ
    """

    def run(self):
        data = self.load("data")
        data = data[["id", "sales", "d"]]
        # for i in range(1, 8):
        #     print("Shifting:", i)
        #     data["lag_" + str(i)] = data.groupby(["id"])["sales"].transform(
        #         lambda x: x.shift(i)
        #     )
        data["lag_7"] = data.groupby(["id"])["sales"].transform(lambda x: x.shift(7))
        data["lag_1"] = data.groupby(["id"])["sales"].transform(lambda x: x.shift(1))
        data = data.drop(columns="sales")
        data = self.set_index(data)
        self.dump(data)


class ShortRollingLag(Feature):
    """
    28日シフトしないlag
    まずは7日のみ
    """

    def run(self):
        data = self.load("data")
        data = data[["id", "sales", "d"]]
        # for i in [14, 30, 60]:
        for i in [7]:
            print("Rolling period:", i)
            data["rolling_mean_" + str(i)] = data.groupby(["id"])["sales"].transform(
                lambda x: x.shift(1).rolling(i).mean()
            )
            data["rolling_std_" + str(i)] = data.groupby(["id"])["sales"].transform(
                lambda x: x.shift(1).rolling(i).std()
            )
        data = data.drop(columns="sales")
        data = self.set_index(data)
        self.dump(data)
