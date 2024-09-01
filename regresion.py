import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

# Streamlit app
st.title('Stock Price Prediction with LSTM')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker:', 'AAPL')

# Set default end date to today and let the user select the start date
end_date = st.date_input('End Date:', datetime.today())
start_date = st.date_input('Start Date:', pd.to_datetime('2020-01-01'))

# Fetch data
if ticker:
    @st.cache
    def fetch_data(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        return data

    data = fetch_data(ticker, start_date, end_date)

    if data.empty:
        st.write("No data available for the given ticker and date range.")
    else:
        st.write(f"Data for {ticker} from {start_date} to {end_date}")
        st.write(data.head())

        # Preprocessing
        data = data[['Close', 'Volume']]

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Prepare the training data
        def create_dataset(dataset, time_step=1):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                a = dataset[i:(i + time_step), :]
                X.append(a)
                y.append(dataset[i + time_step, 0])  # Predicting the 'Close' price
            return np.array(X), np.array(y)

        time_step = 60  # Use last 60 days to predict the next day's price
        X, y = create_dataset(scaled_data, time_step)

        # Reshape input to be [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

        # Split data into training and testing sets
        training_size = int(len(X) * 0.8)
        X_train, X_test = X[:training_size], X[training_size:]
        y_train, y_test = y[:training_size], y[training_size:]

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, X.shape[2])))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Train the model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

        # Prediction
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform the predictions to the original scale
        train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], 1))), axis=1))[:, 0]
        test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], 1))), axis=1))[:, 0]

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['Close'], label='Actual Prices')
        plt.plot(data.index[time_step:len(train_predict)+time_step], train_predict, label='Train Predictions')
        plt.plot(data.index[len(train_predict)+(time_step*2):len(data)-1], test_predict, label='Test Predictions')
        plt.legend()
        plt.title('LSTM Model - Actual vs Predicted Stock Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(plt)

        # Predict future prices for the next 10 days
        future_input = scaled_data[-time_step:]
        future_input = future_input.reshape(1, time_step, 2)
        
        future_predictions = []
        for i in range(10):
            future_pred = model.predict(future_input)
            future_predictions.append(future_pred[0, 0])
            future_input = np.append(future_input[:, 1:, :], [[future_pred, future_input[:, -1, 1]]], axis=1)

        future_predictions = scaler.inverse_transform(np.concatenate((np.array(future_predictions).reshape(-1, 1), np.zeros((10, 1))), axis=1))[:, 0]

        future_dates = [end_date + timedelta(days=i) for i in range(1, 11)]
        future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
        future_df.set_index('Date', inplace=True)

        st.write(f'Predicted Prices for the Next 10 Days:')
        st.write(future_df)

        # Plot future predictions
        plt.figure(figsize=(10, 5))
        plt.plot(future_df.index, future_df['Predicted Close'], marker='o', linestyle='-', color='green', label='Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Predicted Stock Prices for the Next 10 Days')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)
