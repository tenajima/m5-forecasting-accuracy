import gokart
from nyaggle.validation import TimeSeriesSplit


class GetFold(gokart.TaskOnKart):
    def output(self):
        return self.make_target("./fold/fold.pkl")

    def run(self):
        times = [
            (("2012-04-25", "2013-04-25"), ("2013-04-25", "2013-06-30")),
            (("2013-04-25", "2014-04-25"), ("2014-04-25", "2014-06-30")),
            (("2014-04-25", "2015-04-25"), ("2015-04-25", "2015-06-30")),
            (("2015-04-25", "2016-04-25"), ("2016-04-25", "2016-06-30")),
        ]
        fold = TimeSeriesSplit("date", times)
        self.dump(fold)
