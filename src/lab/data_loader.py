import pathlib
import logging
import numpy as np
import pandas as pd

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
                    data = np.concatenate((data, np.array(row[1:]).reshape(1, -1)))

        adjusted_data = pd.read_csv(f'{path}/stock_market_historical_data/prices-split-adjusted.csv')

        return adjusted_data

    def transform(self):  # Todo: delete all the tickers that we wont be using
        pass

    def preprocess(self):
        pass
