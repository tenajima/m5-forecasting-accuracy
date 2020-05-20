import luigi
from dotenv import load_dotenv

# from scripts.feature import GetFeature
from scripts.feature.get_feature import SeveralLagFeature


if __name__ == "__main__":
    load_dotenv("env")
    luigi.build([SeveralLagFeature()], workers=1, local_scheduler=True)
