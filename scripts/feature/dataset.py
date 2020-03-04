import gokart
import luigi
import pandas as pd


class ReadAndTransformData(gokart.TaskOnKart):

    def run(self):
        train = pd.read_csv(
            "../input/m5-forecasting-accuracy/sales_train_validation.csv"
        ).set_index("id")
        predict_columns = ["d_19" + str(i) for i in range(14, 70)]

        for col in predict_columns:
            train[col] = pd.NA

        # d_から始まるやつのみにしぼる
        daily = train[train.columns[train.columns.str.startswith("d_")]].copy()
        dataset = (
            daily.stack().reset_index().rename(columns={"level_1": "d", 0: "target"})
        )

        # calendarのdate情報を付与
        calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv', usecols=['d', 'date'])
        dataset = dataset.merge(calendar, on='d', how='left')

        # dateカラムをdatetime型にしてソートする
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset = dataset.sort_values('date').reset_index(drop=True)

        self.dump(dataset)
