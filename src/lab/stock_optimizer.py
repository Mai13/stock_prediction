from data_loader import DataLoader
from ticker_selector import ChooseTicker
from graph_maker import CreateGraphs
from tecnical_analysis import TechnicalIndicators
import logging

logger = logging.getLogger('Stock Optimizer')


class StockOptimizer:

    def __init__(self, number_of_tickers):
        self.number_of_tickers = number_of_tickers
        self.data_loader = DataLoader()
        self.choose_ticker = ChooseTicker()
        self.graph_maker = CreateGraphs()
        self.technical_analysis = TechnicalIndicators()

    def run(self):

        logger.info('Select tickers')
        tickers = self.choose_ticker.random_tickers(self.number_of_tickers)
        logger.info(f'The selected tickers are {tickers}')
        for ticker in tickers:
            logger.info(f'Starting with ticker {ticker}')
            data = self.data_loader.load(tickers)
            self.graph_maker.plot_adjusted_prices(ticker, data)
            logger.info(
                f'Adjusted data is loaded and graphed, there are {data.shape[0]} data points')
            logger.info(
                f'Divided in Train : {data.shape[0]*0.7}, Validation : {data.shape[0]*0.15}, test : {data.shape[0]*0.15}')
            data = self.technical_analysis.calculate(data)
            logger.info(f'Graph start here')
            self.graph_maker.plot_adjusted_prices(ticker, data)
