import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import yfinance as yf
import streamlit as st
import time
import datetime


ticker = st.sidebar.text_input('Symbol Code' ,'GAIL.NS')
period = st.sidebar.text_input('Period' ,'1y')
interval = st.sidebar.text_input('Interval' ,'1d')
df = yf.download(ticker ,period = period, interval = interval)
data = df.reset_index()
st.write(df)


# Create interactive candlestick chart
fig = go.Figure(data=[go.Candlestick(x=data['Datetime'],
                                     open=data['Open'],
                                     high=data['High'],
                                     low=data['Low'],
                                     close=data['Close'])])

fig.update_layout(title='Interactive Stock Price Chart',
                  xaxis_title='Date',
                  yaxis_title='Price')

# Add range slider for interactivity
fig.update_layout(xaxis_rangeslider_visible=True)
fig.update_layout(width=1200, height=720, plot_bgcolor='black')
# Display interactive chart using Streamlit
st.plotly_chart(fig)

def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out) #creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]) #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True) #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0) #cross validation

    response = [X_train,X_test , Y_train, Y_test , X_lately]
    return response
forecast_col = 'Close'
forecast_out = 5
test_size = 0.2
X_train, X_test, Y_train, Y_test , X_lately = prepare_data(df,forecast_col,forecast_out,test_size); #calling the method were the cross validation and data preperation is in
learner = LinearRegression() #initializing linear regression model

learner.fit(X_train,Y_train) #training the linear regression model

score=learner.score(X_test,Y_test)#testing the linear regression model
forecast= learner.predict(X_lately) #set that will contain the forecasted data
response={}#creting json object
response['test_score']=score
response['forecast_set']=forecast
fr = pd.DataFrame(forecast)
st.write(fr)
st.write(response)
