import math
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
plt.style.use('fivethirtyeight')

data=yf.download('AAPL',start='2012-01-01', end='2019-12-17')
#print(data.head())
print(data.shape)

plt.figure(figsize=(16,8))
plt.title('Close price History')
plt.plot(data['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price USD ($)',fontsize=18)
#plt.show()
#Create a new dataframe with only the close column
data1=data.filter(['Close'])
#convert the dataframe into the numpy array
dataset=data1.values
#get the number of rows to trian model on
training_data_len =math.ceil(len(dataset)*.8)
print(training_data_len)

#scale the data
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
print(scaled_data)

#create the traning dataset
#create the scaled traning dataset
train_data=scaled_data[0:training_data_len, :]
#split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

#convert the x_train and y_train to numpy arrays
x_train,y_train= np.array(x_train), np.array(y_train)
#reshape the data
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)

#Build the LSTM model
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile the model
model.compile(optimizer='adam',loss='mean_squared_error')

#Train the model
model.fit(x_train,y_train,batch_size=1,epochs=1)

#Creating the testing dataset
#crate a new array contaning scaled values from index 1543 to 2003
test_data = scaled_data[training_data_len-60:, :]
#create the data sets x_test and y_test
x_test=[]
y_test=dataset[training_data_len:, :]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

#convert the data into numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
#get the models predicted price vslues
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)
rese = np.sqrt(np.mean(predictions - y_test)**2)
print(rese)
predictions_df = pd.DataFrame(predictions, index=data1.index[-len(predictions):], columns=['Predictions'])

valid=pd.concat([data1[-len(predictions):],predictions_df],axis=1)
#plot the data
train = data[:training_data_len]
Valid = data[training_data_len:]
valid['predictions'] = predictions
#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'], loc='lower right')
plt.show()

#show the valid and predicted price
print(valid)