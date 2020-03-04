from scripts.dataset.read_data import ReadAndTransformData
import gokart
import luigi
from datetime import timedelta

import pandas as pd
from nyaggle.validation import Nth, TimeSeriesSplit

from dataclasses import dataclass


@dataclass
class DataSet:
    history: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame


class GetDatasetOfFold(gokart.TaskOnKart):
    fold_num = luigi.IntParameter()

    def requires(self):
        return ReadAndTransformData()

    def output(self):
        return self.make_target("./dataset/dataset_fold.pkl")

    def run(self):
        data = self.load()

        times = [
            (("2012-04-25", "2013-04-25"), ("2013-04-25", "2013-06-30")),
            (("2013-04-25", "2014-04-25"), ("2014-04-25", "2014-06-30")),
            (("2014-04-25", "2015-04-25"), ("2015-04-25", "2015-06-30")),
        ]
        fold = Nth(self.fold_num, TimeSeriesSplit("date", times))
        train_index, valid_index = next(fold.split(data))

        train = data.iloc[train_index]
        valid = data.iloc[valid_index]

        # 履歴データの期間
        history_start = train["date"].min() - timedelta(days=366)
        history_end = train["date"].min() - timedelta(days=1)

        history = data[(data["date"] >= history_start) & (data["date"] <= history_end)]

        dataset = DataSet(history, train, valid)

        self.dump(dataset)

