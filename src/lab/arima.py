from graph_maker import CreateGraphs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import pathlib
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('ARIMA')


class Arima:

    def __init__(self):

        self.graph_maker = CreateGraphs()
        self.path = f'{pathlib.Path(__file__).parent.parent.absolute()}'

    def __check_stationarity(self, train):

        # Rolling statistics
        roll_mean = train.rolling(30).mean()
        roll_std = train.rolling(5).std()

        # Dickey-Fuller test
        print('Dickey-Fuller test results\n')
        df_test = adfuller(train, regresults=False)
        test_result = pd.Series(df_test[0:4], index=[
            'Test Statistic', 'p-value', '# of lags', '# of obs'])
        print(test_result)
        for k, v in df_test[4].items():
            print('Critical value at %s: %1.5f' % (k, v))

        self.graph_maker.plot_rolling_statistics(
            train, roll_mean, roll_std, self.path)

    def run(self):

        filepath = pathlib.Path(__file__).resolve().parent

        df = pd.read_csv(
            f'{filepath.resolve().parent}/stock_market_historical_data/prices-split-adjusted.csv',
            index_col=0)
        dfa = df[df['symbol'] == 'AAPL']

        dfa.index.sort_values()
        # Convert index to pandas datetime
        dfa.index = pd.to_datetime(dfa.index, format="%Y/%m/%d")
        df_final = dfa.drop(
            ['symbol', 'open', 'low', 'high', 'volume'], axis=1)

        # Conver to Series to run Dickey-Fuller test
        df_final = pd.Series(df_final['close'])

        check_stationarity(df_final)

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
