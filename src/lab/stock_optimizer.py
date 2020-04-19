from data_loader import DataLoader
from ticker_selector import ChooseTicker
from graph_maker import CreateGraphs
from tecnical_analysis import TechnicalIndicators
from feed_forward_nn import FeedForwardNN
import logging

logger = logging.getLogger('Stock Optimizer')


class StockOptimizer:

    def __init__(self, number_of_tickers, train):
        self.number_of_tickers = number_of_tickers
        self.data_loader = DataLoader()
        self.choose_ticker = ChooseTicker()
        self.graph_maker = CreateGraphs()
        self.technical_analysis = TechnicalIndicators()
        self.number_of_past_points = 7
        self.is_train_necesry = train  # boolean

    def run(self):

        logger.info('Select tickers')
        tickers = self.choose_ticker.random_tickers(self.number_of_tickers)
        logger.info(f'The selected tickers are {tickers}')
        for ticker in tickers:
            logger.info(f'Starting with ticker {ticker}')
            data = self.data_loader.load(ticker)
            self.graph_maker.plot_adjusted_prices(ticker, data)
            # logger.info(f'Adjusted data is loaded and graphed, there are {data.shape[0]} data points')
            # logger.info(f'Divided in Train : {data.shape[0]*self.train_set}, Validation : {data.shape[0]*self.validation_set}, test : {data.shape[0]*self.test_set}')
            data = self.technical_analysis.calculate(data)
            # logger.info(f'Graph start here')
            # self.graph_maker.plot_adjusted_prices(ticker, data) # TODO: hay que hacer esto bien
            # self.graph_maker.plot_technical_indicators(data, 100) # TODO: hay
            # que hacer esto bien
            logger.info(f'Technical analysis graphs')

            train, test, validation, train_graph, test_graph, validation_graph = self.data_loader.transform(
                data, neural_net=True, number_of_past_points=self.number_of_past_points)
            self.graph_maker.plot_train_test_val(
                ticker, train_graph, test_graph, validation_graph)
            model = FeedForwardNN(
                dimension_of_first_layer=self.number_of_past_points * train[0][0].shape[1],
                number_of_epoch=10,
                learning_rate_adam=0.01,
                ticker=ticker)
            predictions, true_values, train_loss, val_loss = model.run(
                train, validation, test)
