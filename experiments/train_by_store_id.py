import luigi
from scripts.model.model_by_store_id import TrainAllStoreId
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv("env")
    luigi.build([TrainAllStoreId()], workers=1, local_scheduler=True)
