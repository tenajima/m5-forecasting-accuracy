import pandas as pd
import gokart

from .get_feature import GetFeature

END_TRAIN = 1913
TEST_DAYS = 28
TRAIN_DAYS = 100


class GetTrain(gokart.TaskOnKart):
    def requires(self):
        return {"feature": GetFeature()}

    def output(self):
        return self.make_target(f"./feature/train.pkl", use_unique_id=False)

    def run(self):
        feature = self.load("feature")
        # dをindexからおとしてそれをもとにしてtrainをわける
        train = feature.reset_index(level=1).query(
            f"{END_TRAIN - TRAIN_DAYS} < d <= {END_TRAIN - (TEST_DAYS)}"
        )
        self.dump(train)


class GetValid(gokart.TaskOnKart):
    def requires(self):
        return {"feature": GetFeature()}

    def output(self):
        return self.make_target(f"./feature/valid.pkl", use_unique_id=False)

    def run(self):
        feature = self.load("feature")
        # dをindexからおとしてそれをもとにしてvalidをわける
        valid = feature.reset_index(level=1).query(
            f"{END_TRAIN - TEST_DAYS} < d <= {END_TRAIN}"
        )
        self.dump(valid)


class GetTest(gokart.TaskOnKart):
    def requires(self):
        return {"feature": GetFeature()}

    def output(self):
        return self.make_target(f"./feature/test.pkl", use_unique_id=False)

    def run(self):
        feature = self.load("feature")
        # dをindexからおとしてそれをもとにしてtestをわける
        test = feature.reset_index(level=1).query(f"d > {END_TRAIN}")
        self.dump(test)
