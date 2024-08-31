import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit app
st.title('Stock Price Prediction with Machine Learning')

# User input for stock ticker
ticker = st.text_input('Enter Stock Ticker:', 'AAPL')
start_date = st.date_input('Start Date:', pd.to_datetime('2020-01-01'))
end_date = st.date_input('End Date:', pd.to_datetime('2023-01-01'))

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

        # Feature Engineering
        data['Volume'] = data['Volume'].shift(-1)  # Predicting next day's volume
        data = data.dropna()  # Drop rows with NaN values

        X = data[['Close', 'Volume']]  # Features
        y = data['Close'].shift(-1).dropna()  # Target variable: next day's close price
        X = X[:-1]  # Align X and y

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Model Training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f'Mean Squared Error: {mse:.2f}')

        # Plot results
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Stock Prices')
        st.pyplot(plt)

        # Future Price Prediction
        last_row = data[['Close', 'Volume']].iloc[-1:]
        future_price = model.predict(last_row)
        st.write(f'Predicted Next Day Closing Price: ${future_price[0]:.2f}')
