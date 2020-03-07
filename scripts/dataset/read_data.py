import gokart
import pandas as pd


class ReadAndTransformData(gokart.TaskOnKart):
    def run(self):
        train = pd.read_csv(
            "../input/m5-forecasting-accuracy/sales_train_validation.csv"
        ).set_index("id")
        predict_columns = ["d_19" + str(i) for i in range(14, 70)]

        for col in predict_columns:
            train[col] = -1

        # d_から始まるやつのみにしぼる
        daily = train[train.columns[train.columns.str.startswith("d_")]].copy()
        dataset = (
            daily.reset_index()
            .melt(id_vars=["id"])
            .rename(columns={"variable": "d", "value": "target"})
        )

        # calendarのdate情報を付与
        calendar = pd.read_csv(
            "../input/m5-forecasting-accuracy/calendar.csv", usecols=["d", "date"]
        )
        dataset = dataset.merge(calendar, on="d", how="left")

        # dateカラムをdatetime型にしてソートする
        dataset["date"] = pd.to_datetime(dataset["date"])
        dataset = dataset.sort_values("date").reset_index(drop=True)

        # lightgbmがInt型に対応していない
        dataset["target"] = dataset["target"].astype(int)

        self.dump(dataset)
