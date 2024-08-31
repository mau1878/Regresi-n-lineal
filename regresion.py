import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

# Streamlit app
st.title('Stock Price Prediction with Machine Learning')

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

        # Feature Engineering
        data['Volume'] = data['Volume'].shift(-1)  # Predicting next day's volume
        data = data.dropna()  # Drop rows with NaN values

        X = data[['Close', 'Volume']]  # Features
        y = data['Close'].shift(-1).dropna()  # Target variable: next day's close price
        X = X[:-1]  # Align X and y

        if X.empty or y.empty:
            st.write("Not enough data to train the model.")
        else:
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

            # Predict future prices for the next 10 days
            future_dates = [end_date + timedelta(days=i) for i in range(1, 11)]
            future_data = pd.DataFrame(index=future_dates, columns=['Close', 'Volume'])

            # Initialize future_data with the last known values
            last_row = data[['Close', 'Volume']].iloc[-1:]
            future_data.loc[:, 'Volume'] = last_row['Volume'].values

            # Initialize the first prediction
            future_data.loc[future_dates[0], 'Close'] = model.predict([last_row.values.flatten()])[0]

            # Iteratively predict future prices
            for i in range(1, len(future_dates)):
                # Prepare input data for prediction
                previous_close = future_data.loc[future_dates[i - 1], 'Close']
                previous_volume = future_data.loc[future_dates[i - 1], 'Volume']
                
                # Check for NaN values and handle them
                if pd.isna(previous_close) or pd.isna(previous_volume):
                    st.write(f"Prediction error: NaN values encountered at index {i}.")
                    continue

                # Predict the next day's close price
                try:
                    future_data.loc[future_dates[i], 'Close'] = model.predict([[previous_close, previous_volume]])[0]
                except ValueError as e:
                    st.write(f"Prediction error: {e}")
                    continue

                # Assuming volume remains constant (you can extend this by predicting volume as well)
                future_data.loc[future_dates[i], 'Volume'] = previous_volume

            st.write(f'Predicted Prices for the Next 10 Days:')
            st.write(future_data)

            # Plot future predictions
            plt.figure(figsize=(10, 5))
            plt.plot(future_data.index, future_data['Close'], marker='o', linestyle='-', color='green', label='Predicted Prices')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title('Predicted Stock Prices for the Next 10 Days')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
