import gokart
import luigi

from scripts.model.model_lightgbm import train_lgb
from scripts.train.preprocess import DataForML, Preprocess


class TrainStratifiedKFold(gokart.TaskOnKart):
    random_state = luigi.IntParameter()
    params = luigi.DictParameter()
    model_type = luigi.Parameter()

    def requires(self):
        return Preprocess()

    def output(self):
        return self.make_target("./train/models.pkl")

    def run(self):
        data: DataForML = self.load()
        train_X = data.train_X
        train_y = data.train_y
        fold = data.fold
        groups = data.groups

        if self.model_type == "lgb":
            models = train_lgb(
                fold,
                self.params.get_wrapped(),
                train_X,
                train_y,
                self.random_state,
                groups,
            )
        else:
            raise ValueError(f"{self.model_type} is not defined!")

        self.dump(models)
