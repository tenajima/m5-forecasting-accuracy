import luigi
from dotenv import load_dotenv

from scripts.feature import GetFeature


if __name__ == "__main__":
    load_dotenv("env")
    luigi.build([GetFeature()], workers=1, local_scheduler=True)
