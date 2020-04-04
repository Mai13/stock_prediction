import pandas as pd


class TechnicalIndicators:

    def __init__(self):
        self.ma_short_period = 7
        self.ma_long_period = 21
        self.ema_coef = 0.5

    def __moving_averange(self, data):

        data['ma_short_period'] = data['close'].rolling(window=self.ma_short_period).mean()
        data['ma_long_period'] = data['close'].rolling(window=self.ma_long_period).mean()
        return data

    def __exponential_moving_average(self, data):

        data['ema'] = data['close'].ewm(com=self.ema_coef).mean()
        return data

    def __macd(self, data):

        # TODO: SOLVE THIS INDICATOR

        # data['26ema'] = pd.ewm('close', halflife=self.ma_long_period)
        # data['26ema'] = data['close'].ewm(com=self.ema_coef).mean()
        # data['12ema'] = pd.ewm('close', halflife=self.ma_short_period)
        # data['26ema'] = pd.ewma(data['close'], span=self.ma_long_period)
        # data['12ema'] = pd.ewma(data['close'], span=self.ma_short_period)
        data['MACD'] = (data['12ema'] - data['26ema'])
        data.drop(data['26ema'], inplace=True)
        data.drop(data['12ema'], inplace=True)
        return data

    def __bollinger_bands(self, data):

        data['20sd'] = pd.stats.moments.rolling_std(data['close'], self.ma_long_period)
        data['upper_band'] = data['ma_long_period'] + (data['20sd'] * 2)
        data['lower_band'] = data['ma_long_period'] - (data['20sd'] * 2)
        return data

    def __momentum(self, data):

        data['momentum'] = data['close'] - 1
        return data

    def calculate(self, data):

        data = self.__moving_averange(data)
        data = self.__exponential_moving_average(data)
        data = self.__macd(data)
        data = self.__bollinger_bands(data)
        data = self.__momentum(data)
        return data
