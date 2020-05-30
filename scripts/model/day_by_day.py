from dataclasses import dataclass
import pandas as pd
import lightgbm as lgb
import gokart
import luigi

from scripts.feature.split_feature import GetTrain, GetValid, GetTest, TEST_DAYS

TARGET = "sales"


@dataclass
class Result:
    valid: pd.DataFrame
    test: pd.DataFrame


class TrainAndPredictOneDay(gokart.TaskOnKart):
    target_day = luigi.IntParameter()

    def requires(self):
        return {"train": GetTrain(), "valid": GetValid(), "test": GetTest()}

    def output(self):
        return self.make_target(
            f"./predicts/result_day_{str(self.target_day).zfill(2)}.pkl",
            use_unique_id=False,
        )

    def run(self):
        train = self.load_data_frame("train")
        valid = self.load_data_frame("valid")
        test = self.load_data_frame("test")

        # test期間の何日目のモデルかというのとdのmap
        shift_day_map = {
            i + 1: d for (i, d) in enumerate(test.reset_index()["d"].unique())
        }
        # test のdが何曜日かというmap
        dow_map = test.drop_duplicates("d").set_index("d")["dayofweek"].to_dict()

        d = shift_day_map[self.target_day]
        dow = dow_map[d]
        valid = valid.query(f"dayofweek == {dow}")
        test = test.query(f"d == {d}")

        common_columns = [
            "sell_price",
            "lag_t28",
            "lag_t29",
            "lag_t30",
            "rolling_mean_t7",
            "rolling_mean_t30",
            "rolling_std_t30",
            "rolling_skew_t30",
            "rolling_kurt_t30",
            "rolling_mean_t60",
            "rolling_mean_t90",
            "rolling_std_t90",
            "rolling_mean_t180",
            "rolling_std_t180",
            "price_change_t1",
            "price_change_t365",
            "rolling_price_std_t7",
            "rolling_price_std_t30",
            "snap_CA",
            "snap_TX",
            "snap_WI",
            "wm_yr_wk",
            "dayofweek",
        ]

        id_columns = [
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
        ]

        params = {
            "boosting_type": "gbdt",
            "metric": "rmse",
            "objective": "poisson",
            #         'objective': "tweedie",
            "n_jobs": -1,
            "seed": 110,
            "learning_rate": 0.05,
            "bagging_fraction": 0.75,
            "bagging_freq": 10,
            "colsample_bytree": 0.75,
        }

        if self.target_day < TEST_DAYS:  # 28日目のときはshift_columnsがないから
            feature_columns = train.columns
            shift_column_names = [
                f"shift_{day}" for day in range(self.target_day, TEST_DAYS)
            ]
            # 全部使うと多すぎるので最新と7日ごとのものを使う
            shift_column_names = list(
                set(shift_column_names)
                & set([shift_column_names[0]] + ["shift_7", "shift_14", "shift_21"])
            )

            shift_columns = []
            for col in shift_column_names:
                shift_columns += feature_columns[
                    feature_columns.str.startswith(col)
                ].tolist()
            use_columns = common_columns + id_columns + shift_columns
        elif self.target_day == TEST_DAYS:  # 28日目なら
            use_columns = common_columns + id_columns
        else:
            raise ValueError("something wrong when make use_columns")

        dataset_train = lgb.Dataset(train[use_columns], train[TARGET])
        dataset_valid = lgb.Dataset(valid[use_columns], valid[TARGET])

        model = lgb.train(
            params,
            dataset_train,
            num_boost_round=1000,
            valid_sets=[dataset_train, dataset_valid],
            early_stopping_rounds=200,
            verbose_eval=100,
        )

        valid["pred"] = model.predict(valid[use_columns])
        test[TARGET] = model.predict(test[use_columns])

        result = Result(valid[["d", TARGET, "pred"]], test[["d", TARGET]])

        self.dump(result)


class TrainAndPredictAllDays(gokart.TaskOnKart):
    def requires(self):
        tasks = {
            i + 1: TrainAndPredictOneDay(target_day=i + 1) for i in range(TEST_DAYS)
        }
        return tasks

    def output(self):
        return self.make_target("submit.csv", use_unique_id=False)

    def run(self):
        sample_submit = pd.read_csv(
            "../input/m5-forecasting-accuracy/sample_submission.csv"
        )
        self.dump(sample_submit)
