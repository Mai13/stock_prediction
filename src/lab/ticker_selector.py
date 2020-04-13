import random
import pathlib


class ChooseTicker:

    def __init__(self):
        random.seed(195)
        self.path = pathlib.Path(__file__).parent.parent.parent.absolute()
        print(self.path)

    def modern_portfolio_optimization(self, data):
        pass

    def random_tickers(self, number_of_tickers):

        tickers_list = open(
            f"{self.path}/stock_market_historical_data/tickers_list.csv").read().splitlines()
        return random.sample(tickers_list, number_of_tickers)
