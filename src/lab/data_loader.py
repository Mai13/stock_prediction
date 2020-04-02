import pathlib
import logging
import numpy as np

logger = logging.getLogger('Data loader')
path = pathlib.Path(__file__).parent.parent.parent.absolute()


class DataLoader:

    def load(self, tickers):  # Todo: choose between price and adjusted price
        # read line by line and just load the necessary arrays
        data = np.empty((0, 79))
        with open(f'{path}/stock_market_historical_data/fundamentals.csv') as f:
            for i, line in enumerate(f):
                row = line.split(',')
                ticker = row[1]
                if ticker in tickers:
                    print(data.shape, np.array(row).reshape(1, -1).shape)
                    data = np.concatenate((data, np.array(row).reshape(1, -1)))
        print(data)
        pass

    def transform(self):  # Todo: delete all the tickers that we wont be using
        pass

    def preprocess(self):
        pass
