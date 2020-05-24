import luigi
from dotenv import load_dotenv

# from scripts.feature import GetFeature
from scripts.dataset import ReadAndTransformData

# from scripts.feature.get_feature import SeveralLagFeature


if __name__ == "__main__":
    load_dotenv("env")
    # luigi.build([GetFeature()], workers=1, local_scheduler=True)
    luigi.build([ReadAndTransformData()], workers=1, local_scheduler=True)
