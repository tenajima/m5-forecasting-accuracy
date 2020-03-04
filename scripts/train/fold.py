import gokart
import luigi
from sklearn.model_selection import StratifiedKFold


class GetFold(gokart.TaskOnKart):
    fold_type = luigi.Parameter()
    n_splits = luigi.IntParameter()
    random_state = luigi.IntParameter()
    shuffle = luigi.BoolParameter(default=True)

    def run(self):
        folds = {
            "sf": StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        }

        self.dump(folds[self.fold_type])
