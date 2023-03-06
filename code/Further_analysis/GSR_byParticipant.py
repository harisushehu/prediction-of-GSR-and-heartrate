# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:34:59 2021

@author: harisushehu
"""


#Go to the project directory/path and run  /home/harisushehu/myenv/bin/python GSR_byParticipant.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense     
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

import pandas as pd
import glob
import numpy as np

#scoring imports
from sklearn.metrics import mean_squared_error
from math import sqrt

from tensorflow.python.framework import ops
ops.reset_default_graph() 

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

tf.compat.v1.keras.backend.set_session(sess)


#define model
def model(X, dropout):
            
        #initalize RNN
        regressor = Sequential()
        
        #Add first layer
        regressor.add(LSTM(units = 150, activation='relu', return_sequences = True, input_shape = (X.shape[1], 1)))
        regressor.add(Dropout(dropout)) #dropout = 0.2
      
        regressor.add(Dense(units = 150))
        regressor.add(Dropout(dropout))
        
        #Add fourth layer
        regressor.add(LSTM(units = 75))
        regressor.add(Dropout(dropout))
        
        #Add fifth layer
        regressor.add(Dense(units = 1))
        
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100000,
            decay_rate=0.96,
            staircase=True)
        opt = Adam(learning_rate=lr_schedule)
        regressor.compile(optimizer = opt, loss = 'mean_squared_error', metrics=['mean_squared_error'])
        
        return regressor

def nrmse(rmse, y_test):   
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]


def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset


#Save results
import csv
from csv import writer

#Append data in csv function
def append_list_as_row(file_name, list_of_elem):
   
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)
        

csvFileName = './data/results_GSR_byParticipant.csv'

#read in CSV file
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'w', newline = '') as f:
        
        header = ['Participant', 'RMSE', 'NRMSE', 'LRRMSE', 'LRNRMSE', 'DTRMSE', 'DTNRMSE'] 
        filewriter = csv.DictWriter(f, fieldnames = header)
        filewriter.writeheader()


    
print("Reading data...")

path = 'dataset path' # use your path # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding = 'ISO-8859-1', header = 0)
    li.append(df)
    

dataset = pd.concat(li, axis=0, ignore_index=True)

#drop NaN (missing) features
dataset = dataset.dropna()

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

participant_count = 1
numInit = 0

for countCheck in range(1, 154):
    
    test_fileReader = []
    train_fileReader = []
    
    for normloop in range(1, 154):#153
        
        participant = normloop
        
        flag = False
        video_names = []
        
        for filename in all_files:
            
            if len(str(participant)) == 1:  
                partNo = "00" + str(participant)
                
            elif len(str(participant)) == 2:  
                partNo = "0" + str(participant)
                
            else:
                partNo = str(participant)
                
           
            if partNo in filename:
                
                video_names.append(filename)
                
                flag = True
        
        if flag == True:
               
            if normloop == participant_count:
                
                for iterate in range(0, len(video_names)):
                    
                    test_reader = pd.read_csv(video_names[iterate], encoding = 'ISO-8859-1', header = 0)  
                    test_fileReader.append(test_reader)  
                    
            else:
                
                for iterate in range(0, len(video_names)):
                    train_reader = pd.read_csv(video_names[iterate], encoding = 'ISO-8859-1', header = 0)
                    train_fileReader.append(train_reader)                       
    else:
        print("Participant" + str(normloop) + "does not exists")                
        
    
    test_data = pd.concat(test_fileReader, axis=0, ignore_index=True)    
    train_data = pd.concat(train_fileReader, axis=0, ignore_index=True)
                
    #dropping NaN (missing) features
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
        
    X_train = train_data.iloc[:, train_data.columns != 'GSR_mean']
    y_train = train_data.iloc[:, train_data.columns == 'GSR_mean'].values
    
    X_test = test_data.iloc[:, test_data.columns != 'GSR_mean']
    y_test = test_data.iloc[:, test_data.columns == 'GSR_mean'].values
        
    from sklearn.preprocessing import StandardScaler
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    keras_callbacks   = [
          EarlyStopping(monitor='val_loss',patience=3, min_delta=1e-3,mode='min'),
          ModelCheckpoint("best_GSR_Participantmodel.hdf5", monitor='val_loss', save_best_only=True, mode='min', save_freq='epoch')]
    
    
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    
    #normalize train and test data
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    
    sc1_X = StandardScaler()
    sc1_y = StandardScaler()
    
    X_test = sc1_X.fit_transform(X_test)
    y_test = sc1_y.fit_transform(y_test)
    
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
    
    LinearReg = LinearRegression().fit(X_train, y_train)
    DTReg = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)
    
    LinearPred = LinearReg.predict(X_test)
    DTPred = DTReg.predict(X_test)
    
    Linear_mse = mean_squared_error(y_test, LinearPred)
    Linear_rmse = sqrt(Linear_mse)
    
    DT_mse = mean_squared_error(y_test, DTPred)
    DT_rmse = sqrt(DT_mse)
    
    print("****************************************************")
    print("Results for participant no. " + str(countCheck))
    
    #Evaluate Performance
    print("LR Mean squared error", Linear_mse)
    print("LR Root mean squared error", Linear_rmse)
    
    NLinear_nrmse = nrmse(Linear_rmse, y_test)
    print("LR Normalized root mean squared error", NLinear_nrmse)
    
    print("****************************************************")
    print("Results for participant no. " + str(countCheck))
    
    #Evaluate Performance
    print("DT Mean squared error", DT_mse)
    print("DT Root mean squared error", DT_rmse)
    
    NDT_nrmse = nrmse(DT_rmse, y_test)
    print("DT Normalized root mean squared error", NDT_nrmse)
    
    print("X_train :", X_train.shape)
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
       
    print("After reshaping...")
    print(type(X_train))
    print("X_train :", X_train.shape)
    
    print("X_test :", X_test.shape)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], 1])
       
    print(type(X_test))
    print("X_test :", X_test.shape)  

    print("Evaluating "+ str(countCheck) +" participant...")

    dropouts = 0.2
    regressor = model(X_train, dropouts)
    
    history = regressor.fit(X_train, y_train, epochs = 100, verbose = 1, callbacks=keras_callbacks, batch_size = 256, validation_data = (X_test, y_test))
    
    from tensorflow.keras.models import load_model
    # load best model from single file
    regressor = load_model('best_GSR_Participantmodel.hdf5')
    
    # make predictions
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    
    print("****************************************************")
    print("Results for participant no. " + str(countCheck))
    
    #Evaluate Performance
    print("Mean squared error", mse)
    print("Root mean squared error", rmse)
    
    nrmse_val = nrmse(rmse, y_test)
    print("Normalized root mean squared error", nrmse_val)
    
    #append and save results
    row_contents = [str(countCheck), str(rmse), str(nrmse_val), str(Linear_rmse), str(NLinear_nrmse), str(DT_rmse), str(NDT_nrmse)]
    append_list_as_row(csvFileName, row_contents)   

    predictions = sc1_y.inverse_transform(y_pred)
        
    valid = test_data
    predicted = np.array(predictions)
    all_predictions = predictions     
    valid['Predictions'] = all_predictions
    
    from matplotlib import pyplot as plt 
    plt.clf()
    f, ax = plt.subplots(1)
    plt.title('Model')
    plt.xlabel('Time', fontsize=10)
    plt.ylabel('GSR_mean', fontsize=10)
    #plt.plot(train['GSR_mean'])
    plt.plot(valid[['GSR_mean', 'Predictions']])
    plt.legend(['Test', 'Predictions'], loc='lower right')
    ax.set_ylim(ymin=-3e-5, ymax = 5e-5)
    ax.set_xlim(xmin=0, xmax = 3000)
    #plt.show()
    
    filename = "../data/GSR_byParticipant/Pred_" + str(countCheck)
    plt.savefig(filename)
    
    participant_count = participant_count + 1 #increment participant counter
        
 


