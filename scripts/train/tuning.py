import gokart
import luigi

from scripts.model.model_lightgbm import tune_lgb
from scripts.train.preprocess import DataForML, Preprocess


class TuningLGB(gokart.TaskOnKart):
    random_state = luigi.IntParameter()

    def requires(self):
        return Preprocess()

    def run(self):
        data: DataForML = self.load()
        train_X = data.train_X
        train_y = data.train_y
        result = tune_lgb(train_X, train_y, seed=self.random_state)
        self.dump(result)
