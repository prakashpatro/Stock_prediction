import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2022-04-30'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker (search google for stock tickers)', 'AAPL')

df = data.DataReader(user_input,'yahoo', start, end)

st.subheader('Data from 2010-2022')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
figi = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(figi)

st.subheader('Closing Price vs Time Chart with 100 and 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fige = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r', label = '100MA')
plt.plot(ma200, 'g', label = '200MA')
plt.plot(df.Close, 'b', label = 'Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fige)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])


from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range=(0,1))

#data_training_array = Scaler.fit_transform(data_training)

#Loading Model
model = load_model('prediction_model.h5')

#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = Scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
Scaler =Scaler.scale_

scale_factor = 1/Scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph
st.subheader('Predictions vs Original graph')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)