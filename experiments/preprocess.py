import luigi
from dotenv import load_dotenv

from scripts.train.tuning import TuningLGB
from scripts.train.train import TrainStratifiedKFold
from scripts.submit.predict import Predict

if __name__ == "__main__":
    load_dotenv()
    luigi.run(["Preprocess", "--workers", "8", "--local-scheduler"])
