import luigi
from dotenv import load_dotenv

from scripts.dataset.read_data import ReadAndTransformData


if __name__ == "__main__":
    load_dotenv("env")
    luigi.build([ReadAndTransformData()], workers=1, local_scheduler=True)
