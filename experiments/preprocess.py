from scripts.feature.get_feature import GetFeature
import luigi
from dotenv import load_dotenv

from scripts.feature.split_feature import GetTrain, GetValid, GetTest


if __name__ == "__main__":
    load_dotenv("env")
    # luigi.build([GetFeature()], workers=1, local_scheduler=True)
    # luigi.build([ReadAndTransformData()], workers=1, local_scheduler=True)
    luigi.build([GetTrain(), GetValid(), GetTest()], local_scheduler=False)
