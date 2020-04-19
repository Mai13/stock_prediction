import pathlib
import pandas as pd
import matplotlib.pyplot as plt

filepath = pathlib.Path(__file__).resolve().parent
print()

df_prices = pd.read_csv(f'{filepath.resolve().parent}/stock_market_historical_data/prices.csv')
df_adjusted_prices = pd.read_csv(f'{filepath.resolve().parent}/stock_market_historical_data/prices-split-adjusted.csv')
df_fundamentals = pd.read_csv(f'{filepath.resolve().parent}/stock_market_historical_data/fundamentals.csv')
df_securities = pd.read_csv(f'{filepath.resolve().parent}/stock_market_historical_data/securities.csv')

print(df_prices['symbol'].unique())
print(len(df_prices['symbol'].unique()))
print(df_prices.dtypes)
print(df_prices['date'].max())
print(df_prices['date'].min())

print(df_adjusted_prices['symbol'].unique())
print(len(df_adjusted_prices['symbol'].unique()))
print(df_adjusted_prices.dtypes)
print(df_adjusted_prices['date'].max())
print(df_adjusted_prices['date'].min())

print(len(df_securities['GICS Sector'].unique()))
print(len(df_securities['GICS Sub Industry'].unique()))

# Plotting data by sector

for sector in df_securities['GICS Sector'].unique():
    plt.figure(figsize=(50, 25))
    tickers = df_securities['Ticker symbol'][df_securities['GICS Sector']==sector].unique()
    labels = []
    for ticker in tickers:
        filtered_data = df_prices[df_prices['symbol'] == ticker]
        plt.plot(filtered_data['date'], filtered_data['close'])
        labels.append(ticker)
    plt.legend(labels)
    plt.savefig(f'{filepath}/results/{sector}.png')
    plt.close()
