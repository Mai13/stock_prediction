from data_loader import DataLoader
from ticker_selector import ChooseTicker
from graph_maker import CreateGraphs
from tecnical_analysis import TechnicalIndicators
from feed_forward_nn import FeedForwardNN
from random_forest import RandomForest
from xgboost_model import XGBoost
from arima import Arima
import logging

logger = logging.getLogger('Stock Optimizer')


class StockOptimizer:

    def __init__(self, number_of_tickers, models_and_parameters, overfitting_threshold, number_of_past_points):
        self.number_of_tickers = number_of_tickers
        self.data_loader = DataLoader()
        self.choose_ticker = ChooseTicker()
        self.graph_maker = CreateGraphs()
        self.technical_analysis = TechnicalIndicators()
        self.models_and_parameters = models_and_parameters
        self.overfitting_threshold = overfitting_threshold
        self.number_of_past_points = number_of_past_points

    def run(self):

        logger.info('Select tickers')
        tickers = self.choose_ticker.random_tickers(self.number_of_tickers)
        logger.info(f'The selected tickers are {tickers}')
        # for ticker in tickers:
        for ticker in ['MCHP']:
            logger.info(f'Starting with ticker {ticker}')
            data = self.data_loader.load(ticker)
            self.graph_maker.plot_adjusted_prices(ticker, data)
            data = self.technical_analysis.calculate(data)
            logger.info(f'Technical analysis graphs')
            train, test, validation, train_graph, test_graph, validation_graph = self.data_loader.transform(
                data, number_of_past_points=self.number_of_past_points)
            self.graph_maker.plot_train_test_val(ticker, train_graph, test_graph, validation_graph)

            for position, model_name in enumerate([element.get('model') for element in self.models_and_parameters]):

                if model_name == 'feed_forward_neural_net':

                    model = FeedForwardNN(
                        dimension_of_first_layer=self.number_of_past_points * train[0][0].shape[1],
                        ticker=ticker,
                        overfitting_threshold=self.overfitting_threshold,
                    )
                if model_name == 'random_forest':
                    model = RandomForest(ticker=ticker, overfitting_threshold=self.overfitting_threshold)
                if model_name == 'xgboost':
                    model = XGBoost(ticker=ticker, overfitting_threshold=self.overfitting_threshold)
                if model_name == 'Arima':
                    model = Arima(ticker=ticker, overfitting_threshold=self.overfitting_threshold)
                best_parameters, mse, trend_ratio, prediction, true_values = model.run(
                    train=train, val=validation, test=test, model_parameters=self.models_and_parameters[position])
                logger.info(f'The best scenario for a Feed Forward Neural Net is {best_parameters}, mse: {mse}, ratio of trend {trend_ratio*100}')
                self.graph_maker.plot_test_results(true_values, prediction, ticker, mse, model_name)
