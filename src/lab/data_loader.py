import pathlib
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger('Data loader')
path = pathlib.Path(__file__).parent.parent.parent.absolute()


class DataLoader:

    def load(self, tickers):  # Todo: choose between price and adjusted price
        # read line by line and just load the necessary arrays
        data = np.empty((0, 78))
        with open(f'{path}/stock_market_historical_data/fundamentals.csv') as f:
            for i, line in enumerate(f):
                row = line.strip().split(',')
                ticker = row[1]
                if ticker in tickers:
                    data = np.concatenate(
                        (data, np.array(row[1:]).reshape(1, -1)))

        adjusted_data = pd.read_csv(
            f'{path}/stock_market_historical_data/prices-split-adjusted.csv')
        adjusted_data['date'] = pd.to_datetime(adjusted_data['date'])
        return adjusted_data

    def __transform_data_for_nn(self):

        pass

    # Todo: delete all the tickers that we wont be using
    def transform(self, adjusted_data, neural_net):
        data_array = adjusted_data.values

        return data_array
