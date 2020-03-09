from dataclasses import dataclass
from datetime import timedelta

import gokart
import luigi
import pandas as pd
from nyaggle.validation import Nth

from scripts.dataset.read_data import ReadAndTransformData
from scripts.fold import GetFold


@dataclass
class Dataset:
    history: pd.DataFrame
    data: pd.DataFrame


class GetDataSet(gokart.TaskOnKart):
    def requires(self):
        return {"data": ReadAndTransformData()}

    def output(self):
        return self.make_target("./dataset/dataset.pkl")

    def run(self):
        all_data = self.load("data")
        history = all_data[all_data["date"] < "2016-03-27"]
        data = all_data[all_data["date"] >= "2016-03-27"]

        dataset = Dataset(history, data)

        self.dump(dataset)
