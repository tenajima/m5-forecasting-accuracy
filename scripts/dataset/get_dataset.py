from dataclasses import dataclass

import gokart
import pandas as pd

from scripts.dataset.read_data import ReadAndTransformData


@dataclass
class Dataset:
    history: pd.DataFrame
    data: pd.DataFrame


class GetDataset(gokart.TaskOnKart):
    def requires(self):
        return {"data": ReadAndTransformData()}

    def output(self):
        return self.make_target("./dataset/dataset.pkl")

    def run(self):
        all_data = self.load("data")
        # history = all_data[all_data["date"] < "2016-03-27"]
        # data = all_data[all_data["date"] >= "2016-03-27"]

        dataset = Dataset(
            all_data[all_data["date"] < "2016-03-27"],
            all_data[all_data["date"] >= "2016-03-27"],
        )

        self.dump(dataset)
