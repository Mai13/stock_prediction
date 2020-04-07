import pandas as pd
import numpy as np


class TechnicalIndicators:

    def __init__(self):
        self.ma_short_period = 7
        self.ma_long_period = 21
        self.ema_coef = 0.5
        self.num_of_std = 2

    def __moving_averange(self, data):

        data['ma_short_period'] = data['close'].rolling(
            window=self.ma_short_period).mean()
        data['ma_long_period'] = data['close'].rolling(
            window=self.ma_long_period).mean()
        return data

    def __exponential_moving_average(self, data):

        data['ema'] = data['close'].ewm(com=self.ema_coef).mean()
        return data

    def __macd(self, data):

        # TODO: SOLVE THIS INDICATOR, HAS AN OLD VERSION OF PANDAS

        # data['26ema'] = pd.ewm('close', halflife=self.ma_long_period)
        # data['26ema'] = data['close'].ewm(com=self.ema_coef).mean()
        # data['12ema'] = pd.ewm('close', halflife=self.ma_short_period)
        # data['26ema'] = pd.ewma(data['close'], span=self.ma_long_period)
        # data['12ema'] = pd.ewma(data['close'], span=self.ma_short_period)
        data['12ema'] = data['close'].ewm(span=12, adjust=False).mean()
        data['26ema'] = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = (data['12ema'] - data['26ema'])
        # print(data.columns)
        # print(data.head())
        data.drop('26ema', axis=1, inplace=True)
        data.drop('12ema', axis=1, inplace=True)
        return data

    def __bollinger_bands(self, data):

        rolling_mean = data['close'].rolling(window=self.ma_long_period).mean()
        rolling_std = data['close'].rolling(window=self.ma_long_period).std()
        data['upper_band'] = rolling_mean + (rolling_std * self.num_of_std)
        data['lower_band'] = rolling_mean - (rolling_std * self.num_of_std)

        """
        data['20sd'] = pd.stats.moments.rolling_std(data['close'], self.ma_long_period)
        data['upper_band'] = data['ma_long_period'] + (data['20sd'] * 2)
        data['lower_band'] = data['ma_long_period'] - (data['20sd'] * 2)
        """
        return data

    def __rsi(self, data):

        # short period
        delta = data['close'].diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[self.ma_short_period - 1]] = np.mean(u[:self.ma_short_period])  # first value is sum of avg gains
        u = u.drop(u.index[:(self.ma_short_period - 1)])
        d[d.index[self.ma_short_period - 1]] = np.mean(d[:self.ma_short_period])  # first value is sum of avg losses
        d = d.drop(d.index[:(self.ma_short_period - 1)])
        rs = pd.stats.moments.ewma(u, com=self.ma_short_period - 1, adjust=False) / \
                pd.stats.moments.ewma(d, com=self.ma_short_period - 1, adjust=False)
        data['rsi_short'] = 100 - 100 / (1 + rs)

        # long period
        delta = data['close'].diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        u[u.index[self.ma_long_period - 1]] = np.mean(u[:self.ma_long_period])  # first value is sum of avg gains
        u = u.drop(u.index[:(self.ma_long_period - 1)])
        d[d.index[self.ma_long_period - 1]] = np.mean(d[:self.ma_long_period])  # first value is sum of avg losses
        d = d.drop(d.index[:(self.ma_long_period - 1)])
        rs = pd.stats.moments.ewma(u, com=self.ma_long_period - 1, adjust=False) / \
             pd.stats.moments.ewma(d, com=self.ma_long_period - 1, adjust=False)
        data['rsi_long'] = 100 - 100 / (1 + rs)

        return data

    def calculate(self, data):

        data = self.__moving_averange(data)
        data = self.__exponential_moving_average(data)
        data = self.__macd(data)
        data = self.__bollinger_bands(data)
        data = self.__rsi(data)
        return data
