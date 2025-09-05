import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import ta

#Get data and inspect
#REALISED IT HAS A TICKER COLUMN
#NEED TO RENAME DATE AS A COLUMN AND DS, CLOSE AS y

st.title("AAPL Stock Forecaster using Prophet")
stock_options = ["AAPL", "NVDA", "TSLA", "PLTR"]
ticker = st.selectbox("Enter stock ticker:", stock_options)
period_predictions = st.number_input("Forecast horizon (days):", min_value=1, max_value=365)

if ticker:
    data_frame = yf.download(ticker, start = "2019-01-01", end = "2025-09-01") #read data for a 5 year time frame.
    if not data_frame.empty:
        data_frame = data_frame.reset_index() #SETS DATE AS COLUMN
        data_frame.columns = data_frame.columns.get_level_values(0) #Gets rid of ticker
        #train up to jan 
        training_end = pd.to_datetime("2025-01-01")
        train_data_frame = data_frame[data_frame['Date'] <= training_end].copy()
        test_data_frame = data_frame[data_frame['Date'] > training_end].copy()
        #TRIAL OF RSI
        #RSI FOR ONLY TRAINING AND THEN CONCATENATE THE DATA FRAME
        train_data_frame["rsi"] = ta.momentum.RSIIndicator(train_data_frame['Close'], window = 14).rsi()#2week timeperiod
        train_data_frame = train_data_frame.dropna(subset=["rsi"]).reset_index(drop=True) #GETS RID OF THE FIRST 14 NAN RSI

        combined_close =  pd.concat([train_data_frame['Close'], test_data_frame['Close']])
        rsi_all = ta.momentum.RSIIndicator(combined_close, window=14).rsi()
        test_data_frame['rsi'] = rsi_all.iloc[len(train_data_frame):].reset_index(drop=True)

        train_data_frame = train_data_frame[["Date", "Close", "rsi"]].rename(columns = {"Date": "ds", "Close": "y"}) #Renames column
        test_data_frame = test_data_frame[['Date', 'Close', 'rsi']].rename(columns={'Date': 'ds', 'Close': 'y'})
        train_data_frame['y'] = np.log(train_data_frame['y']) #ADDED
        test_data_frame['y'] = np.log(test_data_frame['y']) #ADDED
        
    else:
        st.warning(f"NO DATA FOUND FOR {ticker}")
else:
    st.warning("Please select a stock ticker")


#actually fit the prophet model
#CREATES AND FITS MODEL
model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05) #add features here\
# model.add_regressor("volume")
model.add_regressor("rsi")
model.fit(train_data_frame)
#.fit obviously uses the generalized additive model
# considers trend, seasonality and the holiday term as well as potential error
#add dummy variables for holiday if provided, fourier series for seasonlity, finds points of changing trend for slope



#FORECASTING, BUT ALSO ADDS THE REGRESSOR OF VOLUME TO THE FORECASTED FRAME,SINCE ADDITIVIVE
future_frame = model.make_future_dataframe(periods = period_predictions)#Forcast for about half a year - 180
# future_frame['volume'] = 0
# future_frame.loc[:len(data_frame)-1, 'volume'] = data_frame['volume'].values
# future_frame.loc[len(data_frame):, 'volume'] = data_frame['volume'].iloc[-1]  
future_frame['rsi'] = 0
future_frame.loc[:len(train_data_frame)-1, 'rsi'] = train_data_frame['rsi'].values
future_frame.loc[len(train_data_frame):, 'rsi'] = train_data_frame['rsi'].iloc[-1]  

forecast = model.predict(future_frame)#makes future data_frame by taking last date and making one new row per date for the next 180 (days by default)
#PLOTTING
forecast['yhat'] = np.exp(forecast['yhat'])
forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(train_data_frame['ds'], np.exp(train_data_frame['y']), label="Actual", color = "blue") #ACTUAL VALUES
ax.plot(forecast['ds'], forecast['yhat'], label="Forecast", color = "orange") #FORECASTED VALUES

ax.plot(test_data_frame['ds'], np.exp(test_data_frame['y']), label="Test Actual", color='green') #ACTUAL TEST VALUES AFTER FORECAST START
forecast_start = pd.to_datetime("2025-01-01")
ax.axvline(forecast_start, color='r', linestyle='--', label="Forecast start") #HIGHLIGHT WHERE FORECAST STARTS
ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2) #INTERVALS
ax.legend()
st.pyplot(fig)

#EVALUATIONS
data_frame_cv = cross_validation(model, initial="730 days", period="180 days", horizon="730 days") #2 year testing, 2 year horizon
data_frame_perf = performance_metrics(data_frame_cv)
actual = np.exp(test_data_frame['y'])
#predictions come from just train  for the first part, and then train + test
predictions = forecast['yhat'].values[len(train_data_frame) :len(train_data_frame)+len(test_data_frame)]

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))
mape = np.mean(np.abs((actual - predictions) / actual)) * 100
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAPE: {mape:.2f}%")



