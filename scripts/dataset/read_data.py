import gokart
import luigi
import pandas as pd

from scripts.utils import reduce_mem_usage


class ReadAndTransformData(gokart.TaskOnKart):
    nrows = luigi.IntParameter()

    def run(self):
        if self.nrows == 0:
            self.nrows = None

        sales_train_validation = pd.read_csv(
            "../input/m5-forecasting-accuracy/sales_train_validation.csv",
            nrows=self.nrows,
        )
        sales_train_validation = pd.melt(
            sales_train_validation,
            id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
            var_name="day",
            value_name="demand",
        )
        print(
            "Melted sales train validation has {} rows and {} columns".format(
                sales_train_validation.shape[0], sales_train_validation.shape[1]
            )
        )
        sales_train_validation = reduce_mem_usage(sales_train_validation)

        submission = pd.read_csv(
            "../input/m5-forecasting-accuracy/sample_submission.csv"
        )
        # seperate test dataframes
        test1_rows = [row for row in submission["id"] if "validation" in row]
        test2_rows = [row for row in submission["id"] if "evaluation" in row]
        test1 = submission[submission["id"].isin(test1_rows)]
        test2 = submission[submission["id"].isin(test2_rows)]

        # change column names
        test1.columns = [
            "id",
            "d_1914",
            "d_1915",
            "d_1916",
            "d_1917",
            "d_1918",
            "d_1919",
            "d_1920",
            "d_1921",
            "d_1922",
            "d_1923",
            "d_1924",
            "d_1925",
            "d_1926",
            "d_1927",
            "d_1928",
            "d_1929",
            "d_1930",
            "d_1931",
            "d_1932",
            "d_1933",
            "d_1934",
            "d_1935",
            "d_1936",
            "d_1937",
            "d_1938",
            "d_1939",
            "d_1940",
            "d_1941",
        ]
        test2.columns = [
            "id",
            "d_1942",
            "d_1943",
            "d_1944",
            "d_1945",
            "d_1946",
            "d_1947",
            "d_1948",
            "d_1949",
            "d_1950",
            "d_1951",
            "d_1952",
            "d_1953",
            "d_1954",
            "d_1955",
            "d_1956",
            "d_1957",
            "d_1958",
            "d_1959",
            "d_1960",
            "d_1961",
            "d_1962",
            "d_1963",
            "d_1964",
            "d_1965",
            "d_1966",
            "d_1967",
            "d_1968",
            "d_1969",
        ]
        product = sales_train_validation[
            ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        ].drop_duplicates()

        # merge with product table
        test2["id"] = test2["id"].str.replace("_evaluation", "_validation")
        test1 = test1.merge(product, how="left", on="id")
        test2 = test2.merge(product, how="left", on="id")
        test2["id"] = test2["id"].str.replace("_validation", "_evaluation")

        #
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

        sales_train_validation["part"] = "train"
        test1["part"] = "test1"
        test2["part"] = "test2"

        data = pd.concat([sales_train_validation, test1, test2], axis=0)

        del sales_train_validation, test1, test2

        # drop some calendar features
        calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
        calendar = reduce_mem_usage(calendar)
        calendar.drop(["weekday", "wday", "month", "year"], inplace=True, axis=1)

        # delete test2 for now
        data = data[data["part"] != "test2"]

        data = pd.merge(data, calendar, how="left", left_on=["day"], right_on=["d"])
        data.drop(["d", "day"], inplace=True, axis=1)
        # get the sell price data (this feature should be very important)
        sell_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
        sell_prices = reduce_mem_usage(sell_prices)
        data = data.merge(
            sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left"
        )
        print(
            "Our final dataset to train has {} rows and {} columns".format(
                data.shape[0], data.shape[1]
            )
        )

        self.dump(data)
