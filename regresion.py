import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

st.title('Stock Price Prediction using LSTM')

# User inputs for stock ticker and date range
ticker = st.text_input('Enter Stock Ticker:', 'AAPL')
start_date = st.date_input('Start Date', pd.to_datetime('2015-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

# Load stock data
data = yf.download(ticker, start=start_date, end=end_date)
st.write(f"Loaded {len(data)} rows of data for {ticker}.")

# Display the last few rows of the data
st.write(data.tail())

# Feature selection: using 'Close' and 'Volume' to predict 'Close'
data = data[['Close', 'Volume']]

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create a function to prepare the dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Hyperparameters
time_step = 60  # 60 days of historical data to predict the next day

# Preparing the training and test datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape the input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 2)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1)

# Predicting the stock prices
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual prices
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], 1))), axis=1))[:, 0]

# Plotting
plt.figure(figsize=(12, 6))

# Actual Prices
plt.plot(data.index, data['Close'], label='Actual Prices')

# Train Predictions
train_index = data.index[time_step:len(train_predict) + time_step]
plt.plot(train_index, train_predict, label='Train Predictions')

# Test Predictions
test_index = data.index[len(train_predict) + (time_step * 2):-1]

# Adjust the length of test_predict to match test_index
test_predict = test_predict[:len(test_index)]

plt.plot(test_index, test_predict, label='Test Predictions')

plt.legend()
plt.title('LSTM Model - Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
st.pyplot(plt)

# Predict future prices for the next 10 days
n_future_days = 10
last_data = scaled_data[-time_step:]
future_predictions = []

for _ in range(n_future_days):
    pred = model.predict(last_data.reshape(1, time_step, 2))[0, 0]
    future_predictions.append(pred)
    last_data = np.append(last_data[1:], [[pred, last_data[-1, 1]]], axis=0)

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((n_future_days, 1))), axis=1))[:, 0]

# Display future predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_future_days, freq='B')
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
st.write("Future 10-Day Predictions:")
st.write(future_df)

# Plot future predictions
plt.figure(figsize=(10, 5))
plt.plot(future_df['Date'], future_df['Predicted Close'], marker='o', label='Future Predictions')
plt.legend()
plt.title('LSTM Model - Future 10-Day Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
st.pyplot(plt)
