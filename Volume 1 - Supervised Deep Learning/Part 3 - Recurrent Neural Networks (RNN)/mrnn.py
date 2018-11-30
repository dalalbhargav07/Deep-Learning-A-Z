# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:46:09 2018

@author: bdalal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#---------- Part 1 Data PreProcessing-----------
# Importing the data set
data = pd.read_csv("Google_Stock_Price_Train.csv")
training_data = data.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
scaled_training_data = sc.fit_transform(training_data)

# Creatuiing a data structure with 60 timestamps and 1 output
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(scaled_training_data[i-60:i, 0])
    y_train.append(scaled_training_data[i,0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshaping as 3dtensor as RNN expects input like that
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

#---------- Part 2 Building a RNN -----------
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Initialize the RNN
regressor = Sequential()

# Adding first LSTM layer with Dropout Regularization
regressor.add(LSTM(units=50, return_sequences= True, input_shape= (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding Second LSTM layer with Dropout Regularization
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

# Adding Third LSTM layer with Dropout Regularization
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

# Adding Fourth/last LSTM layer with Dropout Regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)