from graph_maker import CreateGraphs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pathlib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('ARIMA')


class Arima:

    def __init__(self, ticker):

        self.graph_maker = CreateGraphs()
        self.path = f'{pathlib.Path(__file__).parent.parent.absolute()}'
        self.ticker = ticker

    def __transform_data(self, dataset):

        x = np.empty((0, 77))  # todo: set as parameter
        y = np.empty((0, 1))
        for pos, data in enumerate(dataset):
            x = np.vstack((x, data[0].flatten().reshape(1, -1)))
            y = np.vstack((y, np.array(data[1])))

        return x, y

    def __check_stationarity(self, train):

        # Rolling statistics
        roll_mean = pd.DataFrame(train).rolling(30).mean()
        roll_std = pd.DataFrame(train).rolling(5).std()

        # Dickey-Fuller test
        print('Dickey-Fuller test results\n')
        df_test = adfuller(pd.DataFrame(train), regresults=False)
        test_result = pd.Series(df_test[0:4], index=[
            'Test Statistic', 'p-value', '# of lags', '# of obs'])
        print(test_result)
        for k, v in df_test[4].items():
            print('Critical value at %s: %1.5f' % (k, v))

        self.graph_maker.plot_rolling_statistics(train, roll_mean, roll_std, self.path)

    def run(self, train, val, test, model_parameters):

        train_x, train_y = self.__transform_data(train)
        self.__check_stationarity(train_y)

        # Log transform time series
        df_final_log = np.log(df_final)
        df_final_log.dropna(inplace=True)
        check_stationarity(df_final_log)
        # Log Differencing
        df_final_log_diff = df_final_log - df_final_log.shift()
        df_final_log_diff.dropna(inplace=True)
        check_stationarity(df_final_log_diff)
        # Differencing
        df_final_diff = df_final - df_final.shift()
        df_final_diff.dropna(inplace=True)
        check_stationarity(df_final_diff)

        from statsmodels.tsa.stattools import acf, pacf

        df_acf = acf(df_final_diff)
        df_pacf = pacf(df_final_diff)

        fig1 = plt.figure(figsize=(20, 10))
        ax1 = fig1.add_subplot(211)
        fig1 = sm.graphics.tsa.plot_acf(df_acf, ax=ax1)
        ax2 = fig1.add_subplot(212)
        fig1 = sm.graphics.tsa.plot_pacf(df_pacf, ax=ax2)

        model = ARIMA(df_final_diff, (1, 1, 0))
        fit_model = model.fit(full_output=True)
        predictions = model.predict(fit_model.params, start=1760, end=1769)
        fit_model.summary()

        fit_model.predict(start=1760, end=1769)

        pred_model_diff = pd.Series(fit_model.fittedvalues, copy=True)
        pred_model_diff.head()

        # Calculate cummulative sum of the fitted values (cummulative sum of
        # differences)
        pred_model_diff_cumsum = pred_model_diff.cumsum()
        pred_model_diff_cumsum.head()

        # Element-wise addition back to original time series
        df_final_trans = df_final.add(pred_model_diff_cumsum, fill_value=0)
        # Last 5 rows of fitted values
        df_final_trans.tail()

        # Last 5 rows of original time series
        df_final.tail()

        # Plot of orignal data and fitted values
        plt.figure(figsize=(20, 10))
        plt.plot(df_final, color='black', label='Original data')
        plt.plot(df_final_trans, color='red', label='Fitted Values')
        plt.legend()
        plt.show()

        x = df_final.values
        y = df_final_trans.values

        # Trend of error
        plt.figure(figsize=(20, 8))
        plt.plot((x - y), color='red', label='Delta')
        plt.axhline((x - y).mean(), color='black', label='Delta avg line')
        plt.legend()
        plt.show()

        import statsmodels.api as sm
