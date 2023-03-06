# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 18:16:05 2021

@author: harisushehu
"""

#Run by running the below command

#Go to the project directory/path and run  /home/harisushehu/myenv/bin/python GSR.py


#Standard package imports
import pandas as pd
import glob
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

#scoring imports
from math import sqrt
from sklearn.metrics import mean_squared_error



from tensorflow.python.framework import ops
ops.reset_default_graph() 

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


tf.compat.v1.keras.backend.set_session(sess)


def nrmse(rmse, y_test):
    
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    
    return nrmse[0]



#define model
def model(X, dropout):
            
        #initalize RNN
        regressor = Sequential()
        
        #Add first layer
        regressor.add(LSTM(units = 150, return_sequences = True, input_shape = (X.shape[1], 1)))
        regressor.add(Dropout(dropout)) #dropout = 0.2
        
        #Add second layer
        regressor.add(LSTM(units = 150, return_sequences = True))
        regressor.add(Dropout(dropout))       
        
        #Add third layer
        regressor.add(Dense(units = 150))
        regressor.add(Dropout(dropout))
        
        #Add fourth layer
        regressor.add(LSTM(units = 150))
        regressor.add(Dropout(dropout))
        
      
        #Add fifth layer
        regressor.add(Dense(units = 1))
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])
        
        return regressor


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

csvFileName = '../data/results_GSR_shift25.csv'

#read in CSV file
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'w', newline = '') as f:
        
        header = ['Fold', 'RMSE', 'NRMSE'] 
        filewriter = csv.DictWriter(f, fieldnames = header)
        filewriter.writeheader()


print("Reading data...")

#path = '../Clean_Data' # use your path # use your path
path = 'dataset path' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding = 'ISO-8859-1', header = 0)
    li.append(df)
    

#move GSR to last column
df = df[[c for c in df if c not in ['GSR_mean']] + ['GSR_mean']]

#Read selected features
feature_names = '../data/FS_DT_pos_best.csv' #path the selected features
features = pd.read_csv(feature_names, encoding = 'ISO-8859-1', header = 0) 

column_names = []
for i in range(0, 260):
    
    if float(features['pos'][i]) > 0.5:
        
        column_names.append(df.columns[i])


#Add label
column_names.append('GSR_mean')
  
print("Reading data...")

#path = '../Clean_Data' # use your path # use your path
path = 'dataset path' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, encoding = 'ISO-8859-1', header = 0, usecols = column_names)
    li.append(df)
    

dataset = pd.concat(li, axis=0, ignore_index=True)

#replace NaN with 0
dataset = dataset.fillna(0)

print("Evaluating...")
        

X = dataset.iloc[:, df.columns != 'GSR_mean']
y = dataset.iloc[:, df.columns == 'GSR_mean'].values

print("X is :", X.shape)
print("y is :", y.shape)



print("Reading data for preprocessing...")

path = 'dataset path' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))

psysio = []
label = []

count = 0
numInit = 0

full_list = []
first_part = []
second_part = []
third_part = []
fourth_part = []
fifth_part = []


for normloop in range(1, 154): #154 --> participants no. (missing samples are disregarded)
    
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
            reader = pd.read_csv(filename, encoding = 'ISO-8859-1', header = 0, usecols = column_names)
            lines= len(reader)
            increament = increament + lines
            
            fileReader.append(reader)
            
            flag = True
    
    if flag == True:
        
        full_list.extend(fileReader)  
        
        count = count + 1
        
    if (count/29) == 1:
        
        first_part.extend(full_list)
        full_list = []
        
    elif (count/29) == 2:
        
        second_part.extend(full_list)
        full_list = []
        
    elif (count/29) == 3:
        
        third_part.extend(full_list)
        full_list = []
        
    elif (count/29) == 4:
        
        fourth_part.extend(full_list)
        full_list = []
        
    elif (count/29) == 5:
        
        fifth_part.extend(full_list)
        full_list = []
   

list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]
for i in range(len(list_of_lists)):
    
    print("Splitting test and train...")
    train_lists = list_of_lists[0:i] + list_of_lists[i+1:]
    test_list = list_of_lists[i]
  
    
    print("Splitting train and eval...")
    second_list_of_lists = train_lists
    for j in range(0,1):
        
        if i < len(list_of_lists)-1:
            train_lists = second_list_of_lists[0:j] + second_list_of_lists[j+1:]
            val_list = second_list_of_lists[i]
            
        else:
            
            train_lists = second_list_of_lists[0:j] + second_list_of_lists[j+1:] 
            val_list = second_list_of_lists[j]
           
            
    
            
    import itertools
    train = list(itertools.chain(*train_lists))
    
        
    train_data = pd.concat(train, axis=0, ignore_index=True)
            
    #replace NaN with 0
    train_data = train_data.fillna(0)
    '''  
    train_X = train_data.iloc[:, df.columns != 'GSR_mean']
    train_y = train_data.iloc[:, df.columns == 'GSR_mean'].values
    
    
    #shift psysio and labels with n rows
    train_X = train_X.iloc[:-25, :]
    train_y = train_y[25:]  
    
    
    train_X['GSR_mean'] = train_y
    '''
    X_train = train_data.iloc[:, df.columns != 'GSR_mean']
    y_train = train_data.iloc[:, df.columns == 'GSR_mean'].values
    
    #scale train                
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
        
    #normalize train X and y
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    test_data = pd.concat(test_list, axis=0, ignore_index=True)
            
    #replace NaN with 0
    test_data = test_data.fillna(0)
    '''   
    test_X = test_data.iloc[:, df.columns != 'GSR_mean']
    test_y = test_data.iloc[:, df.columns == 'GSR_mean'].values
    
    
    #shift psysio and labels with n rows
    test_X = test_X.iloc[:-25, :]
    test_y = test_y[25:]     
  
    
    test_X['GSR_mean'] = test_y
    '''
    X_test = test_data.iloc[:, df.columns != 'GSR_mean']
    y_test = test_data.iloc[:, df.columns == 'GSR_mean'].values
    
    #scale test                
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    #normalize test X and y
    X_test = scaler_X.fit_transform(X_test)
    y_test = scaler_y.fit_transform(y_test)
    
    val_data = pd.concat(val_list, axis=0, ignore_index=True)
            
    #replace NaN with 0
    val_data = val_data.fillna(0)
    
    '''      
    val_X = val_data.iloc[:, df.columns != 'GSR_mean']
    val_y = val_data.iloc[:, df.columns == 'GSR_mean'].values
    
  
    #shift psysio and labels with n rows
    val_X = val_X.iloc[:-25, :]
    val_y = val_y[25:]     
   
    
    val_X['GSR_mean'] = val_y
    '''
    X_val = val_data.iloc[:, df.columns != 'GSR_mean']
    y_val = val_data.iloc[:, df.columns == 'GSR_mean'].values
    
    #scale val              
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    #normalize test X and y
    X_val = scaler_X.fit_transform(X_val)
    y_val = scaler_y.fit_transform(y_val)    

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM
    from tensorflow.keras.layers import Dense     
    from tensorflow.keras.layers import Dropout
  
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], 1])
    
    X_val = np.array(X_val)
    X_val = np.reshape(X_val, [X_val.shape[0], X_val.shape[1], 1])
           
    print("Shapes of X_train, X_test, and X_val...")
    print("X_train :", X_train.shape)
    print("X_test :", X_test.shape)
    print("X_val :", X_val.shape)
    
    #Training...       
    dropouts = 0.2
    regressor = model(X_train, dropouts)
    regressor.fit(X_train, y_train, epochs = 1, verbose = 1, batch_size = 64, validation_data = (X_val, y_val))

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print("Results with 2-layers dropouts of: ", dropouts)
    
    print("MSE of predicted test set: ", mse)
    rmse = sqrt(mse)
    
    print("****************************************************")
    print("Results for participant no. " + str(i))
    
    #Evaluate Performance
    print("Root mean squared error", rmse)
    
    nrmse_val = nrmse(rmse, y_test)
    print("Normalized root mean squared error", nrmse_val)
        
        
    row_contents = [str(i),str(rmse), str(nrmse_val)]
    #Append a list as new line to an old csv file
    append_list_as_row(csvFileName, row_contents)   
   



    
    
    