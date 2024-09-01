import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

st.title('Stock Price Prediction using LSTM')

# User inputs for stock ticker and date range
ticker = st.text_input('Enter Stock Ticker:', 'AAPL')
start_date = st.date_input('Start Date', pd.to_datetime('2015-01-01'))
end_date = st.date_input('End Date', pd.to_datetime('today'))

# Load stock data with error handling
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error(f"No data found for {ticker} from {start_date} to {end_date}. Please check the ticker symbol and date range.")
        st.stop()
except Exception as e:
    st.error(f"Error downloading data: {e}")
    st.stop()

st.write(f"Loaded {len(data)} rows of data for {ticker}.")
st.write(data.tail())

# Feature selection: using 'Close' and 'Volume' to predict 'Close'
data = data[['Close', 'Volume']]

# Data preprocessing
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to prepare the dataset for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# User-adjustable time_step parameter
time_step = st.slider('Select Time Step (number of days):', min_value=30, max_value=120, value=60)

# Check for enough data
if len(data) < time_step + 1:
    st.error(f"Not enough data to train the model with a time step of {time_step} days.")
    st.stop()

# Prepare training and test datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size, :]
test_data = scaled_data[train_size:, :]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Ensure the data is properly shaped and not empty
if X_train.size == 0 or X_test.size == 0:
    st.error("Training or testing data is empty after reshaping. Please check the time step and the amount of data available.")
    st.stop()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 2)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping callback with validation
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with validation split
history = model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=1, validation_split=0.1, callbacks=[early_stop])

# Check if training history is None (which should not happen with the current setup)
if history is None:
    st.error("Model training failed. Please review the data and model parameters.")
    st.stop()

# Display model summary and training loss
st.subheader('Model Architecture')
st.text(model.summary())

st.subheader('Training Loss')
st.line_chart(history.history['loss'])
st.line_chart(history.history['val_loss'])

# Predicting stock prices
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse scaling
train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], 1))), axis=1))[:, 0]
test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], 1))), axis=1))[:, 0]

# Plotting actual vs predicted prices
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Actual Prices')

train_index = data.index[time_step:len(train_predict) + time_step]
plt.plot(train_index, train_predict, label='Train Predictions')

# Adjust test_index and test_predict lengths to avoid mismatch
test_index = data.index[len(train_predict) + (time_step * 2):-1]
min_length = min(len(test_index), len(test_predict))
test_index = test_index[:min_length]
test_predict = test_predict[:min_length]

plt.plot(test_index, test_predict, label='Test Predictions')

plt.legend()
plt.title(f'{ticker} Stock Prices: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
st.pyplot(plt)

# Predicting future prices for the next n days
n_future_days = st.slider('Select Number of Days to Predict into the Future:', min_value=1, max_value=30, value=10)

last_data = scaled_data[-time_step:]
future_predictions = []

for _ in range(n_future_days):
    pred = model.predict(last_data.reshape(1, time_step, 2), verbose=0)[0, 0]
    future_predictions.append(pred)
    last_data = np.append(last_data[1:], [[pred, last_data[-1, 1]]], axis=0)

# Inverse scaling for future predictions
future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((n_future_days, 1))), axis=1))[:, 0]

# Display future predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=n_future_days, freq='B')
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions})

st.subheader(f'{ticker} Predicted Close Prices for the Next {n_future_days} Days')
st.write(future_df)

plt.figure(figsize=(12, 6))
plt.plot(future_df['Date'], future_df['Predicted Close Price'], label='Future Predictions', color='orange')
plt.legend()
plt.title(f'{ticker} Future Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Predicted Price (USD)')
st.pyplot(plt)
