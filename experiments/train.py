import luigi
from dotenv import load_dotenv

from scripts.model.day_by_day import TrainAndPredictAllDays, TrainAndPredictOneDay

if __name__ == "__main__":
    load_dotenv("env")
    luigi.build([TrainAndPredictAllDays()], local_scheduler=True)
