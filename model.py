import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import ta


st.title("AAPL Stock Forecaster using Prophet")
stock_options = ["AAPL", "NVDA", "TSLA", "PLTR"]
ticker = st.selectbox("Enter stock ticker:", stock_options)
period_predictions = st.number_input("Forecast horizon (days):", min_value=1, max_value=365)

if ticker:
    data_frame = yf.download(ticker, start = "2019-01-01", end = "2025-09-01") #read data for a 5 year time frame.
    if not data_frame.empty:
        #PREPROCESSING 
        data_frame = data_frame.reset_index() #Sets the date as a column rather than a row
        data_frame.columns = data_frame.columns.get_level_values(0) #Gets rid of ticker column
        training_date = pd.to_datetime("2025-01-01")
        train_data_frame = data_frame[data_frame['Date'] <= training_date].copy() #CURRENTLY THERE IS NO TEST TRAIN SPLIT BUT THERE WILL BE
        test_data_frame = data_frame[data_frame['Date'] > training_date].copy()

        #RSI CALCULATION - TRAINING
        train_data_frame["rsi"] = ta.momentum.RSIIndicator(train_data_frame['Close'], window = 14).rsi() #Create RSI column
        train_data_frame = train_data_frame.dropna(subset=["rsi"]).reset_index(drop=True) #GETS RID OF THE FIRST 14 NAN RSI

        #RSI CALCULATION - TESTING
        combined = pd.concat([train_data_frame['Close'], test_data_frame['Close']])
        rsi_all = ta.momentum.RSIIndicator(combined, window=14).rsi()
        test_data_frame['rsi'] = rsi_all.iloc[len(train_data_frame):].reset_index(drop=True)

        train_data_frame = train_data_frame[["Date", "Close", "rsi"]].rename(columns = {"Date": "ds", "Close": "y"}) #Renames columns
        test_data_frame = test_data_frame[["Date", "Close", "rsi"]].rename(columns = {"Date": "ds", "Close": "y"}) 
        train_data_frame['y'] = np.log(train_data_frame['y']) #calculates based on log(y) for stability 
        test_data_frame['y'] = np.log(test_data_frame['y'])

        #CREATING MODEL
        # NOTE: Experiment with values and fourier
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05) 
        model.add_regressor("rsi") #Adds rsi as a regressor and fits data to the model, consider mcad
        model.fit(train_data_frame)

        #PREDICTIONS
        future_frame = model.make_future_dataframe(periods = len(test_data_frame)) #previously input predictions
        future_frame['rsi'] = 0
        #creates RSI for the future frame as well
        future_frame.loc[:len(train_data_frame)-1, 'rsi'] = train_data_frame['rsi'].values 
        future_frame.loc[len(train_data_frame):, 'rsi'] = train_data_frame['rsi'].iloc[-1]  
        forecast = model.predict(future_frame)

        #POST PROCESSING
        forecast['yhat'] = np.exp(forecast['yhat'])
        forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
        forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

        #PLOTTING
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(train_data_frame['ds'], np.exp(train_data_frame['y']), label="Training Actual", color = "blue")
        ax.plot(test_data_frame['ds'], np.exp(test_data_frame['y']), label='Test Actual', color='green')
        ax.plot(forecast['ds'], forecast['yhat'], label="Forecast", color = "orange")  #TRAINING AND TEST PREDICTIONS
        forecast_start = pd.to_datetime("2025-01-01")
        ax.axvline(forecast_start, color='r', linestyle='--', label="Forecast start") 
        ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2) 
        ax.legend()
        st.pyplot(fig)

        #EVALUATION
        data_frame_cv = cross_validation(model, initial="730 days", period="180 days", horizon="730 days") 
        data_frame_perf = performance_metrics(data_frame_cv)
        actual = np.exp(test_data_frame['y']) #prev was train
        # predictions = forecast['yhat'].values[:len(train_data_frame)]
        predictions_train = forecast['yhat'].values[:len(train_data_frame)]
        predictions_test = forecast['yhat'].values[len(train_data_frame): len(train_data_frame) + len(test_data_frame)]

        mae = mean_absolute_error(actual, predictions_test)
        rmse = np.sqrt(mean_squared_error(actual, predictions_test))
        mape = np.mean(np.abs((actual - predictions_test) / actual)) * 100
        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2f}%")

    else:
        st.warning(f"NO DATA FOUND FOR {ticker}")
else:
    st.warning("Please select a stock ticker")





