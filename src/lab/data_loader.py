import pathlib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger('Data loader')
path = pathlib.Path(__file__).parent.parent.parent.absolute()


class DataLoader:

    def load(self, ticker):  # Todo: choose between price and adjusted price
        # read line by line and just load the necessary arrays
        adjusted_data = pd.read_csv(
            f'{path}/stock_market_historical_data/prices-split-adjusted.csv')
        adjusted_data['date'] = pd.to_datetime(adjusted_data['date'])
        adjusted_data = adjusted_data[adjusted_data['symbol'] == ticker]
        return adjusted_data

    def __transform_data_for_nn(self, data_array, number_of_past_points):
        transformed_data = []
        for position in range(1, data_array.shape[0] - number_of_past_points):
            transformed_data.append([data_array[position: position + number_of_past_points, :].astype(
                float), int(data_array[position - 1, 3])])  # closing price
        print(data_array.shape)
        train, test_and_val = train_test_split(
            transformed_data, test_size=0.7, random_state=1, shuffle=False)
        validation, test = train_test_split(
            test_and_val, test_size=0.5, random_state=1, shuffle=False)
        return data_array

    # Todo: delete all the tickers that we wont be using
    def transform(self, adjusted_data, neural_net, number_of_past_points):
        adjusted_data.date = [int(one_date.strftime("%s"))
                              for one_date in adjusted_data.date]
        adjusted_data.drop('symbol', axis=1, inplace=True)
        adjusted_data.sort_values(by=['date'], inplace=True, ascending=False)
        adjusted_data.dropna(inplace=True)
        for column in
        print(adjusted_data.columns)
        data_array = adjusted_data.values
        if neural_net:
            data_array = self.__transform_data_for_nn(
                data_array, number_of_past_points)
        return data_array
