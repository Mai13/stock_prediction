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
        data['12ema'] = data['close'].ewm(span=12, adjust=False).mean()
        data['26ema'] = data['close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = (data['12ema'] - data['26ema'])
        data.drop('26ema', axis=1, inplace=True)
        data.drop('12ema', axis=1, inplace=True)
        return data

    def __bollinger_bands(self, data):

        rolling_mean = data['close'].rolling(window=self.ma_long_period).mean()
        rolling_std = data['close'].rolling(window=self.ma_long_period).std()
        data['upper_band'] = rolling_mean + (rolling_std * self.num_of_std)
        data['lower_band'] = rolling_mean - (rolling_std * self.num_of_std)

        return data

    def __rsi(self, data):
        delta = data['close'].diff().dropna()
        window = 15
        up_days = delta.copy()
        up_days[delta <= 0] = 0.0
        down_days = abs(delta.copy())
        down_days[delta > 0] = 0.0
        RS_up = up_days.rolling(window).mean()
        RS_down = down_days.rolling(window).mean()
        data['rsi'] = 100 - 100 / (1 + RS_up / RS_down)

        """

        # short period
        delta = data['close'].diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        # first value is sum of avg gains
        u[u.index[self.ma_short_period - 1]
          ] = np.mean(u[:self.ma_short_period])
        u = u.drop(u.index[:(self.ma_short_period - 1)])
        # first value is sum of avg losses
        d[d.index[self.ma_short_period - 1]
          ] = np.mean(d[:self.ma_short_period])
        d = d.drop(d.index[:(self.ma_short_period - 1)])
        rs = u.ewm(com=self.ma_short_period - 1, adjust=False) / \
            u.ewm(d, com=self.ma_short_period - 1, adjust=False)
        data['rsi_short'] = 100 - 100 / (1 + rs)

        # long period
        delta = data['close'].diff().dropna()
        u = delta * 0
        d = u.copy()
        u[delta > 0] = delta[delta > 0]
        d[delta < 0] = -delta[delta < 0]
        # first value is sum of avg gains
        u[u.index[self.ma_long_period - 1]] = np.mean(u[:self.ma_long_period])
        u = u.drop(u.index[:(self.ma_long_period - 1)])
        # first value is sum of avg losses
        d[d.index[self.ma_long_period - 1]] = np.mean(d[:self.ma_long_period])
        d = d.drop(d.index[:(self.ma_long_period - 1)])
        rs = pd.stats.moments.ewma(u,
                                   com=self.ma_long_period - 1,
                                   adjust=False) / pd.stats.moments.ewma(d,
                                                                         com=self.ma_long_period - 1,
                                                                         adjust=False)
        data['rsi_long'] = 100 - 100 / (1 + rs)

        """

        return data

    def calculate(self, data):
        data.sort_values(by=['date'], inplace=True)
        data = self.__moving_averange(data)
        data = self.__exponential_moving_average(data)
        data = self.__macd(data)
        data = self.__bollinger_bands(data)
        # data = self.__rsi(data) # TODO: This needs to be calculetd right
        # there is some pandas problem
        return data
