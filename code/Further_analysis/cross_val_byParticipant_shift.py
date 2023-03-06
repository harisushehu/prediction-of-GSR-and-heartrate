# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 21:32:09 2021

@author: harisushehu
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense     
from tensorflow.keras.layers import Dropout
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

import pandas as pd
import glob
import numpy as np


#scoring imports
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics

#import keras.backend.tensorflow_backend as KTF
#tf.disable_v2_behavior()

from tensorflow.python.framework import ops
ops.reset_default_graph() 

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(sess)

tf.compat.v1.keras.backend.set_session(sess)

#Save results

import csv
from csv import writer

#Append data in csv function

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
csvFileName = '../data/results_heartrate_cv_participant_shift5.csv'

#read in CSV file
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'w', newline = '') as f:
        
        header = ['Participant', 'MSE', 'RMSE'] 
        filewriter = csv.DictWriter(f, fieldnames = header)
        filewriter.writeheader()

#define model
def model(X):
            
        #initalize RNN
        regressor = Sequential()
        
        #Add first layer
        regressor.add(LSTM(units = 150, return_sequences = True, input_shape = (X.shape[1], 1)))
        regressor.add(Dropout(0.2))
        
        #Add second layer
        regressor.add(LSTM(units = 150, return_sequences = True))
        regressor.add(Dropout(0.2))
        
        
        #Add third layer
        #regressor.add(LSTM(units = 150, return_sequences = True))
        #regressor.add(Dropout(0.2))
        
        
        regressor.add(Dense(units = 150))
        regressor.add(Dropout(0.2))
        
        #Add fifth layer
        regressor.add(LSTM(units = 150))
        regressor.add(Dropout(0.2))
        
        #regressor.add(Dense(units = 150))
        #regressor.add(Dropout(0.2))
        
        #Add fifth layer
        regressor.add(Dense(units = 1))
        
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])
        
        return regressor

    
print("Reading data...")

path = 'dataset path' # use your path # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding = 'ISO-8859-1', header = 0)
    li.append(df)
    

dataset = pd.concat(li, axis=0, ignore_index=True)

#replace NaN with 0
dataset = dataset.fillna(0)

print("Evaluating...")
        

X = dataset.iloc[:, df.columns != 'GSR_mean']
y = dataset.iloc[:, df.columns == 'GSR_mean'].values

print("X is :", X.shape)
print("y is :", y.shape)



print("Reading data for shifting and normalization by participants...")
path = 'dataset path' # use your path # use your path
all_files = sorted(glob.glob(path + "/*.csv"))

psysio = []
label = []


numInit = 0
for normloop in range(1, 154):
    
    participant = normloop
    
    flag = False
    fileReader = []
    
    increament = 0
    for filename in all_files:
        
        if len(str(participant)) == 1:  
            partNo = "00" + str(participant)
            
        elif len(str(participant)) == 2:  
            partNo = "0" + str(participant)
            
        else:
            partNo = str(participant)
            
       
        if partNo in filename: 
            reader = pd.read_csv(filename, encoding = 'ISO-8859-1', header = 0)
            lines= len(reader)
            increament = increament + lines
            
            fileReader.append(reader)
            
            flag = True
       
    if flag == True:
        
        data = pd.concat(fileReader, axis=0, ignore_index=True)
            
        #replace NaN with 0
        data = data.fillna(0)
            
        X = data.iloc[:, df.columns != 'GSR_mean']
        y = data.iloc[:, df.columns == 'GSR_mean'].values
        
        
        #shift psysio and labels with X rows
        X = X.iloc[:-5, :]
        y = y[5:]
        
        from sklearn.preprocessing import MinMaxScaler
            
        sc_X = MinMaxScaler(feature_range = [0, 1])
        sc_y = MinMaxScaler(feature_range = [0, 1])
        
        #normalize X and y
        X = sc_X.fit_transform(X)
        y = sc_y.fit_transform(y)
        
        print("X :", X.shape)
        X = np.array(X)
        X = np.reshape(X, [X.shape[0], X.shape[1], 1])
           
        print("After reshaping...")
        print(type(X))
        print("X :", X.shape)
    
        print("Evaluating "+ str(normloop) +" participant...")
       
        #estimator = KerasRegressor(build_fn=model(X), epochs=1, batch_size=64, verbose=0)
        #kfold = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        #scorers = ['neg_mean_absolute_error']
        
        
        from sklearn.model_selection import KFold
        
        res = []
       
        # prepare cross validation
        kfold = KFold(3)
        # enumerate splits
        for train_index, test_index in kfold.split(X):
            
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
         
            
            regressor = model(X_train)
            regressor.fit(X_train, y_train, epochs = 5, verbose = 1, batch_size = 32, validation_data = (X_test, y_test))
        
            y_pred = regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            
            #append results
            res.append(mse)
    
        #find average mse of kfold    
        mse_avg = statistics.mean(res)
        
        rmse = sqrt(mse_avg)
        
        print("#############****************************************************##############")
        print("Results for participant no. " + str(normloop))
        
        #Evaluate Performance
        print("Mean squared error", mse_avg)
        print("Root mean squared error", rmse)
            
        row_contents = [str(normloop), str(mse_avg),str(rmse)]
        # Append a list as new line to an old csv file
        append_list_as_row(csvFileName, row_contents)   
        
    else:
            
        print("Participant does not exists")     


'''
df = pd.read_csv("../LSTM_NN/results_heartrate_Noshift.csv", encoding = 'ISO-8859-1')


y_test = df["Actual"]
y_pred = df["Predicted"]


from matplotlib import pyplot as plt

plt.plot(y_test[0:50], color = "red", label = "Actual Heart Rate")
plt.plot(y_pred[0:50], color = "blue", label = "Predicted Heart Rate")
#plt.xlim(0,1)
#plt.ylim(0,1)
plt.title("Heart Rate Prediction")
plt.xlabel("EEG Channels")
plt.ylabel("Heart Rate")
plt.legend()
plt.show()

'''
    