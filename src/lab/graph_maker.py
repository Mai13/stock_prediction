import matplotlib.pyplot as plt
import logging
import pathlib
import numpy as np

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
        dataset = data.iloc[-last_days:, :]
        dataset.index = dataset['date']
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

    def plot_train_test_val(self, ticker, train, test, val):

        train_dates = np.empty((0, 1))
        train_closing_prices = np.empty((0, 1))

        for X, y in train:
            train_dates = np.append(train_dates, X[-1])
            train_closing_prices = np.append(train_closing_prices, y)

        test_dates = np.empty((0, 1))
        test_closing_prices = np.empty((0, 1))

        for X, y in test:
            test_dates = np.append(test_dates, X[-1])
            test_closing_prices = np.append(test_closing_prices, y)

        val_dates = np.empty((0, 1))
        val_closing_prices = np.empty((0, 1))

        for X, y in val:
            val_dates = np.append(val_dates, X[-1])
            val_closing_prices = np.append(val_closing_prices, y)

        plt.figure(figsize=(20, 10))
        plt.plot(
            train_dates,
            train_closing_prices,
            label='Training points',
            c='blue')
        plt.plot(
            test_dates,
            test_closing_prices,
            label='Testing points',
            c='orange')
        plt.plot(
            val_dates,
            val_closing_prices,
            label='Validation points',
            c='yellow')
        plt.legend(loc="upper left")
        plt.savefig(f'{self.path}/results/train_test_val_{ticker}.png')
        plt.close()

    def plot_overfitting_graph(
            self,
            train_loss,
            val_loss,
            epochs,
            optimizer,
            learning_rate,
            ticker):

        if train_loss and val_loss:

            plt.figure(figsize=(20, 10))
            plt.plot(
                range(epochs),
                train_loss,
                label='Training points',
                c='blue')
            plt.plot(
                range(epochs),
                val_loss,
                label='Validation points',
                c='orange')
            plt.savefig(
                f'{self.path}/results/overfitting_graph_{ticker}_{optimizer}_{epochs}_{learning_rate}.png')
            plt.close()

    def plot_test_graph(
            self,
            predictions,
            true_values,
            epochs,
            optimizer,
            learning_rate,
            ticker):

        plt.figure(figsize=(20, 10))
        plt.plot(
            range(
                len(true_values)),
            true_values,
            label='Training points',
            c='blue')
        plt.plot(
            range(
                len(predictions)),
            predictions,
            label='Validation points',
            c='orange')
        plt.savefig(
            f'{self.path}/results/test_graph_{ticker}_{optimizer}_{epochs}_{learning_rate}.png')
        plt.close()

    def plot_rolling_statistics(
            self,
            train,
            rolling_mean,
            rolling_statistics,
            path):

        # Plot rolling statistics
        plt.figure(figsize=(20, 10))
        plt.plot(train, color='black', label='Original Data')
        plt.plot(rolling_mean, color='red', label='Rolling Mean(30 days)')
        plt.legend()
        plt.subplot(212)
        plt.plot(
            rolling_statistics,
            color='green',
            label='Rolling Std Dev(5 days)')
        plt.legend()
        plt.savefig(f'{path}/results/ARIMA_rolling_statistics.png')

    def plot_test_results(self, test_set, predictions, ticker, mse, model):

        plt.figure(figsize=(20, 10))
        plt.title(f'{model} of {ticker} with mse {mse}')
        plt.plot(range(0, len(test_set)), test_set, color='blue', label='Original Data')
        plt.plot(range(0, len(predictions)), predictions, color='orange', label='Prediction')
        plt.legend()
        plt.savefig(f'{self.path}/results/{ticker}_{model}.png')
