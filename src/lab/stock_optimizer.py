from data_loader import DataLoader
from ticker_selector import ChooseTicker
import logging

logger = logging.getLogger('Stock Optimizer')


class StockOptimizer:

    def __init__(self, number_of_tickers):
        self.number_of_tickers = number_of_tickers
        self.data_loader = DataLoader()
        self.choose_ticker = ChooseTicker()

    def run(self):
        logger.info('Select tickers')
        tickers = self.choose_ticker.random_tickers(self.number_of_tickers)
        logger.info(f'The selected tickers are {tickers}')
        self.data_loader.load(tickers)
        logger.info('Data for all the tickers is loaded')
