from data_loader import DataLoader
from ticker_selector import ChooseTicker


class StockOptimizer:

    def __init__(self, number_of_tickers):
        self.number_of_tickers = number_of_tickers
        self.data_loader = DataLoader()
        self.choose_ticker = ChooseTicker()

    def run(self):
        tickers = self.choose_ticker.random_tickers(self.number_of_tickers)
        self.data_loader.load(tickers)
