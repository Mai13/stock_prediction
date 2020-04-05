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
        plt.plot(data['date'], data['close'].astype(float))
        plt.savefig(f'{self.path}/results/evolution_of_{ticker}.png')
        plt.close()

    def plot_technical_indicators(self, data, last_days):

        plt.figure(figsize=(16, 10), dpi=100)
        shape_0 = data.shape[0]
        xmacd_ = shape_0 - last_days

        data.sort_values(by=['date'], inplace=True)
        print(len(data.index))
        dataset = data.iloc[-last_days:, :]
        dataset.index = dataset['date']
        print(len(dataset.index))
        x_ = range(3, dataset.shape[0])
        x_ = list(dataset.index)

        # Plot first subplot
        plt.subplot(2, 1, 1)
        plt.plot(
            dataset['ma_short_period'],
            label='MA 7',
            color='g',
            linestyle='--')
        plt.plot(dataset['close'], label='Closing Price', color='b')
        plt.plot(
            dataset['ma_long_period'],
            label='MA 21',
            color='r',
            linestyle='--')
        plt.plot(dataset['upper_band'], label='Upper Band', color='c')
        plt.plot(dataset['lower_band'], label='Lower Band', color='c')
        plt.fill_between(
            x_,
            dataset['lower_band'],
            dataset['upper_band'],
            alpha=0.35)
        plt.title(
            'Technical indicators for Goldman Sachs - last {} days.'.format(last_days))
        plt.ylabel('USD')
        plt.legend()
        """
        # Plot second subplot
        plt.subplot(2, 1, 2)
        plt.title('MACD')
        plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
        plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
        plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
        plt.plot(
            dataset['log_momentum'],
            label='Momentum',
            color='b',
            linestyle='-')

        plt.legend()
        plt.show()
        """
        plt.savefig(f'{self.path}/results/evolution_of_.png')
