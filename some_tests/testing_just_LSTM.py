import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import sklearn
# from sklearn.cross_validation import  train_test_split
import time #helper libraries
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from numpy import newaxis
import pathlib

filepath = pathlib.Path(__file__).resolve().parent

prices_dataset = pd.read_csv(f'{filepath.resolve().parent}/stock_market_historical_data/prices.csv', header=0)

yahoo = prices_dataset[prices_dataset['symbol']=='YHOO']
yahoo_stock_prices = yahoo.close.values.astype('float32')
yahoo_stock_prices = yahoo_stock_prices.reshape(1762, 1)
print(yahoo_stock_prices.shape)

plt.plot(yahoo_stock_prices)
plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))
yahoo_stock_prices = scaler.fit_transform(yahoo_stock_prices)

train_size = int(len(yahoo_stock_prices) * 0.80)
test_size = len(yahoo_stock_prices) - train_size
train, test = yahoo_stock_prices[0:train_size,:], yahoo_stock_prices[train_size:len(yahoo_stock_prices),:]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('compilation time : ', time.time() - start)

model.fit(
    trainX,
    trainY,
    batch_size=128,
    nb_epoch=10,
    validation_split=0.05)


def plot_results_multiple(predicted_data, true_data, length):
    plt.plot(scaler.inverse_transform(true_data.reshape(-1, 1))[length:])
    plt.plot(scaler.inverse_transform(np.array(predicted_data).reshape(-1, 1))[length:])
    plt.show()


# predict lenght consecutive values from a real one
def predict_sequences_multiple(model, firstValue, length):
    prediction_seqs = []
    curr_frame = firstValue

    for i in range(length):
        predicted = []

        print(model.predict(curr_frame[newaxis, :, :]))
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])

        curr_frame = curr_frame[0:]
        curr_frame = np.insert(curr_frame[0:], i + 1, predicted[-1], axis=0)

        prediction_seqs.append(predicted[-1])

    return prediction_seqs

from sklearn.metrics import mean_squared_error
from math import sqrt
## No entiendo bien cómo poder validar el modelo
predict_length = 5
mean_sqrt = []
print('------------')
for i in range(0, len(testX)):
    # print(i)
    predictions = predict_sequences_multiple(model, testX[i], predict_length)
    prediction_0 = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[0]
    test_0 = scaler.inverse_transform(np.array(testY[0]).reshape(-1, 1))[0]
    # print(prediction_0, test_0)
    # print(sqrt(mean_squared_error(test_0, prediction_0)))
    mean_sqrt.append(sqrt(mean_squared_error(test_0, prediction_0)))
print(sum(mean_sqrt)/len(mean_sqrt))
print(scaler.inverse_transform(np.array(predictions).reshape(-1, 1)))
plot_results_multiple(predictions, testY, predict_length)