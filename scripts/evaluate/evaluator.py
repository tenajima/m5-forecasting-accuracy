import gc

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from scripts.utils import reduce_mem_usage
from scipy import sparse

NUM_ITEMS = 30490
DAYS_PRED = 28


class Evaluator:
    def __init__(self, load=False, debug=True):
        def encode_categorical(df, cols):
            for col in cols:
                # Leave NaN as it is.
                le = LabelEncoder()
                df[col] = df[col].fillna("nan")
                df[col] = pd.Series(le.fit_transform(df[col]), index=df.index)
            return df

        def weight_calc(weight_mat_csr, data, product):

            # calculate the denominator of RMSSE, and calculate the weight base on sales amount

            sales_train_val = pd.read_csv(
                "../input/m5-forecasting-accuracy/sales_train_validation.csv"
            )

            d_name = ["d_" + str(i + 1) for i in range(1913)]

            sales_train_val = weight_mat_csr * sales_train_val[d_name].values

            # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
            # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
            df_tmp = (sales_train_val > 0) * np.tile(
                np.arange(1, 1914), (weight_mat_csr.shape[0], 1)
            )

            start_no = np.min(np.where(df_tmp == 0, 9999, df_tmp), axis=1) - 1

            flag = (
                np.dot(
                    np.diag(1 / (start_no + 1)),
                    np.tile(np.arange(1, 1914), (weight_mat_csr.shape[0], 1)),
                )
                < 1
            )

            sales_train_val = np.where(flag, np.nan, sales_train_val)

            # denominator of RMSSE / RMSSEの分母
            weight1 = np.nansum(np.diff(sales_train_val, axis=1) ** 2, axis=1) / (
                1913 - start_no
            )

            # calculate the sales amount for each item/level
            df_tmp = data[
                (data["date"] > "2016-03-27") & (data["date"] <= "2016-04-24")
            ]
            df_tmp["amount"] = df_tmp["demand"] * df_tmp["sell_price"]
            df_tmp = df_tmp.groupby(["id"])["amount"].apply(np.sum)
            df_tmp = df_tmp[product.id].values

            weight2 = weight_mat_csr * df_tmp

            weight2 = weight2 / np.sum(weight2)

            del sales_train_val
            gc.collect()

            return weight1, weight2

        if load:
            print("loadするぜ")
            self.data = pd.read_pickle("./evaluator_data.pkl")
            self.weight_mat_csr = sparse.load_npz("./evaluator_weight_mat_csr.npz")
            self.weight1 = np.load("./evaluator_weight1.npy", allow_pickle=True)
            self.weight2 = np.load("./evaluator_weight2.npy", allow_pickle=True)
        else:
            print("read_csv中")
            calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
            sell_prices = pd.read_csv(
                "../input/m5-forecasting-accuracy/sell_prices.csv"
            )
            sales_train_val = pd.read_csv(
                "../input/m5-forecasting-accuracy/sales_train_validation.csv"
            )
            submission = pd.read_csv(
                "../input/m5-forecasting-accuracy/sample_submission.csv"
            )

            print("encode中")
            # encode for memory
            calendar = encode_categorical(
                calendar,
                ["event_name_1", "event_type_1", "event_name_2", "event_type_2"],
            ).pipe(reduce_mem_usage)

            sales_train_val = encode_categorical(
                sales_train_val,
                ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            ).pipe(reduce_mem_usage)

            sell_prices = encode_categorical(sell_prices, ["item_id", "store_id"]).pipe(
                reduce_mem_usage
            )

            product = sales_train_val[
                ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
            ].drop_duplicates()

            # to remove data before first non-zero demand date, replace these demand as np.nan.
            d_name = ["d_" + str(i + 1) for i in range(1913)]
            sales_train_val_values = sales_train_val[d_name].values

            # calculate the start position(first non-zero demand observed date) for each item / 商品の最初の売上日
            # 1-1914のdayの数列のうち, 売上が存在しない日を一旦0にし、0を9999に置換。そのうえでminimum numberを計算
            tmp = np.tile(np.arange(1, 1914), (sales_train_val_values.shape[0], 1))
            df_tmp = (sales_train_val_values > 0) * tmp
            start_no = np.min(np.where(df_tmp == 0, 9999, df_tmp), axis=1) - 1
            flag = np.dot(np.diag(1 / (start_no + 1)), tmp) < 1
            sales_train_val_values = np.where(flag, np.nan, sales_train_val_values)

            sales_train_val[d_name] = sales_train_val_values
            del tmp, sales_train_val_values

            sales_train_val = pd.melt(
                sales_train_val,
                id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                var_name="day",
                value_name="demand",
            )

            if debug:
                nrows = 365 * 2 * NUM_ITEMS
                sales_train_val = sales_train_val.iloc[-nrows:, :]

            print("data計算中")
            sales_train_val = sales_train_val[~sales_train_val["demand"].isnull()]

            # submission fileのidのvalidation部分と, ealuation部分の名前を取得
            test1_rows = [row for row in submission["id"] if "validation" in row]
            test2_rows = [row for row in submission["id"] if "evaluation" in row]

            # submission fileのvalidation部分をtest1, ealuation部分をtest2として取得
            test1 = submission[submission["id"].isin(test1_rows)]
            test2 = submission[submission["id"].isin(test2_rows)]

            # test1, test2の列名の"F_X"の箇所をd_XXX"の形式に変更
            test1.columns = ["id"] + [f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]
            test2.columns = ["id"] + [f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]

            # test2のidの'_evaluation'を置換
            test2["id"] = test2["id"].str.replace("_evaluation", "_validation")

            # idをキーにして, idの詳細部分をtest1, test2に結合する.
            test1 = test1.merge(product, how="left", on="id")
            test2 = test2.merge(product, how="left", on="id")

            # test1, test2をともにmelt処理する.（売上数量:demandは0）
            test1 = pd.melt(
                test1,
                id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                var_name="day",
                value_name="demand",
            )

            test2 = pd.melt(
                test2,
                id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                var_name="day",
                value_name="demand",
            )

            # validation部分と, evaluation部分がわかるようにpartという列を作り、 test1,test2のラベルを付ける。
            sales_train_val["part"] = "train"
            test1["part"] = "test1"
            test2["part"] = "test2"

            # sales_train_valとtest1, test2の縦結合.
            data = pd.concat([sales_train_val, test1, test2], axis=0)

            # memoryの開放
            del sales_train_val, test1, test2

            # delete test2 for now(6/1以前は, validation部分のみ提出のため.)
            data = data[data["part"] != "test2"]

            # drop some calendar features(不要な変数の削除:weekdayやwdayなどはdatetime変数から後ほど作成できる。)
            calendar.drop(["weekday", "wday", "month", "year"], inplace=True, axis=1)
            # notebook crash with the entire dataset (maybee use tensorflow, dask, pyspark xD)(dayとdをキーにdataに結合)
            data = pd.merge(data, calendar, how="left", left_on=["day"], right_on=["d"])
            data.drop(["d", "day"], inplace=True, axis=1)
            # memoryの開放
            del calendar

            # sell priceの結合
            # get the sell price data (this feature should be very important)
            data = data.merge(
                sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
            )
            print(
                "Our final dataset to train has {} rows and {} columns".format(
                    data.shape[0], data.shape[1]
                )
            )
            # memoryの開放
            del sell_prices

            self.data = reduce_mem_usage(data)
            self.data.to_pickle("evaluator_data.pkl")

            print("weight計算中")
            weight_mat = np.c_[
                np.ones([NUM_ITEMS, 1]).astype(np.int8),  # level 1
                pd.get_dummies(product.state_id.astype(str), drop_first=False)
                .astype("int8")
                .values,
                pd.get_dummies(product.store_id.astype(str), drop_first=False)
                .astype("int8")
                .values,
                pd.get_dummies(product.cat_id.astype(str), drop_first=False)
                .astype("int8")
                .values,
                pd.get_dummies(product.dept_id.astype(str), drop_first=False)
                .astype("int8")
                .values,
                pd.get_dummies(
                    product.state_id.astype(str) + product.cat_id.astype(str),
                    drop_first=False,
                )
                .astype("int8")
                .values,
                pd.get_dummies(
                    product.state_id.astype(str) + product.dept_id.astype(str),
                    drop_first=False,
                )
                .astype("int8")
                .values,
                pd.get_dummies(
                    product.store_id.astype(str) + product.cat_id.astype(str),
                    drop_first=False,
                )
                .astype("int8")
                .values,
                pd.get_dummies(
                    product.store_id.astype(str) + product.dept_id.astype(str),
                    drop_first=False,
                )
                .astype("int8")
                .values,
                pd.get_dummies(product.item_id.astype(str), drop_first=False)
                .astype("int8")
                .values,
                pd.get_dummies(
                    product.state_id.astype(str) + product.item_id.astype(str),
                    drop_first=False,
                )
                .astype("int8")
                .values,
                np.identity(NUM_ITEMS).astype(np.int8),  # item :level 12
            ].T

            self.weight_mat_csr = sparse.csr_matrix(weight_mat)
            sparse.save_npz("evaluator_weight_mat_csr", self.weight_mat_csr)
            del weight_mat

            self.weight1, self.weight2 = weight_calc(
                self.weight_mat_csr, self.data, product
            )
            np.save("evaluator_weight1", self.weight1)
            np.save("evaluator_weight2", self.weight2)

    def wrmsse(self, y_true, preds):
        # this function is calculate for last 28 days to consider the non-zero demand period

        # actual obserbed values / 正解ラベル

        print("y_true, preds用意")
        y_true = y_true[-(NUM_ITEMS * DAYS_PRED) :]
        preds = preds[-(NUM_ITEMS * DAYS_PRED) :]

        # number of columns
        num_col = DAYS_PRED

        # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
        print("reshapeするよ")
        reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
        reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T

        print("sparse matrix")
        train = self.weight_mat_csr * np.c_[reshaped_preds, reshaped_true]

        print("score計算")
        score = np.sum(
            np.sqrt(
                np.mean(np.square(train[:, :num_col] - train[:, num_col:]), axis=1)
                / self.weight1
            )
            * self.weight2
        )

        return score

    def feval(self, preds, data):
        # this function is calculate for last 28 days to consider the non-zero demand period

        # actual obserbed values / 正解ラベル
        y_true = data.get_label()

        y_true = y_true[-(NUM_ITEMS * DAYS_PRED) :]
        preds = preds[-(NUM_ITEMS * DAYS_PRED) :]

        # number of columns
        num_col = DAYS_PRED

        # reshape data to original array((NUM_ITEMS*num_col,1)->(NUM_ITEMS, num_col) ) / 推論の結果が 1 次元の配列になっているので直す
        reshaped_preds = preds.reshape(num_col, NUM_ITEMS).T
        reshaped_true = y_true.reshape(num_col, NUM_ITEMS).T

        train = self.weight_mat_csr * np.c_[reshaped_preds, reshaped_true]

        score = np.sum(
            np.sqrt(
                np.mean(np.square(train[:, :num_col] - train[:, num_col:]), axis=1)
                / self.weight1
            )
            * self.weight2
        )

        return "wrmsse", score, False
