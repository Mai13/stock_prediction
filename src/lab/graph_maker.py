import matplotlib.pyplot as plt
import logging
import pathlib
import pandas as pd

logger = logging.getLogger('Stock Optimizer')


class CreateGraphs:

    def __init__(self):
        self.path = pathlib.Path(__file__).parent.parent.absolute()

    def plot_adjusted_prices(self, ticker, data):

        plt.figure(figsize=(50, 10))
        plt.plot(pd.to_datetime(data['date']), data['close'].astype(float))
        plt.savefig(f'{self.path}/results/evolution_of_{ticker}.png')
        plt.close()
