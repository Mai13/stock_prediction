import pathlib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger('Data loader')
path = pathlib.Path(__file__).parent.parent.parent.absolute()


class DataLoader:

    def __init__(self):
        self.path = pathlib.Path(__file__).parent.parent.absolute()

    def load(self, ticker):  # Todo: choose between price and adjusted price
        # read line by line and just load the necessary arrays
        adjusted_data = pd.read_csv(
            f'{path}/stock_market_historical_data/prices-split-adjusted.csv')
        adjusted_data['date'] = pd.to_datetime(adjusted_data['date'])
        adjusted_data = adjusted_data[adjusted_data['symbol'] == ticker]
        return adjusted_data

    def __transform_data_for_nn(self, data_array, number_of_past_points):
        transformed_data = []
        transformed_data_for_graphs = []
        for position in range(0, data_array.shape[0] - number_of_past_points):
            transformed_data.append([data_array[position: position + number_of_past_points, 1:].astype(
                float), float(data_array[position - 1, 2])])  # closing price
            transformed_data_for_graphs.append([data_array[position: position + number_of_past_points, 0].astype(
                float), float(data_array[position - 1, 2])])

        test_and_val, train = train_test_split(transformed_data, test_size=0.7, random_state=1, shuffle=False)
        test, validation = train_test_split(test_and_val, test_size=0.5, random_state=1, shuffle=False)
        test_and_val_graph, train_graph = train_test_split(transformed_data_for_graphs, test_size=0.7, random_state=1, shuffle=False)
        test_graph, validation_graph = train_test_split(test_and_val_graph, test_size=0.5, random_state=1, shuffle=False)

        return train, test, validation, train_graph, test_graph, validation_graph

    def __scale(self, data):
        data = MinMaxScaler().fit_transform(data.reshape(-1, 1))
        data = StandardScaler().fit_transform(data)
        return data

    def __plot(self, date, closing, name):

        plt.figure(figsize=(20, 10))
        plt.plot(date, closing, label='Training points', c='blue')
        plt.savefig(f'{self.path}/results/{name}.png')
        plt.close()

    # Todo: delete all the tickers that we wont be using
    def transform(self, adjusted_data, neural_net, number_of_past_points):
        adjusted_data.date = [int(one_date.strftime("%s"))
                              for one_date in adjusted_data.date]
        adjusted_data.drop('symbol', axis=1, inplace=True)
        adjusted_data.sort_values(by=['date'], inplace=True, ascending=False)
        adjusted_data.dropna(inplace=True)
        self.__plot(adjusted_data['date'], adjusted_data['close'], 'before_adjust')
        for column_pos in range(1, adjusted_data.shape[1]):
            adjusted_data.iloc[:, column_pos] = self.__scale(adjusted_data.iloc[:, column_pos].values)
        self.__plot(adjusted_data['date'], adjusted_data['close'], 'after_adjust')

        data_array = adjusted_data.values
        if neural_net:
            train, test, validation, train_graph, test_graph, validation_graph = self.__transform_data_for_nn(
                data_array, number_of_past_points)
        return train, test, validation, train_graph, test_graph, validation_graph
