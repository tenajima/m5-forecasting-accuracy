import luigi
from dotenv import load_dotenv

from scripts.model.day_by_day import TrainAndPredictAllDays

if __name__ == "__main__":
    load_dotenv("env")
    luigi.build([TrainAndPredictAllDays()], local_scheduler=False)
