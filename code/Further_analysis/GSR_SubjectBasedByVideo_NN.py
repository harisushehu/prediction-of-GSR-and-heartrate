# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:02:41 2022

@author: harisushehu
"""

#Go to the project directory/path and run  /home/harisushehu/myenv/bin/python GSR_SubjectBySubject.py

import pandas as pd
import glob
import numpy as np
import os
from math import sqrt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import Adam

from tensorflow.python.framework import ops
ops.reset_default_graph() 

import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense     
from tensorflow.keras.layers import Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


tf.compat.v1.keras.backend.set_session(sess)


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


def get_median(ls):
    # sort the list
    ls_sorted = ls.sort()
    # find the median
    if len(ls) % 2 != 0:
        # total number of values are odd
        # subtract 1 since indexing starts at 0
        m = int((len(ls)+1)/2 - 1)
        return ls[m]
    else:
        m1 = int(len(ls)/2 - 1)
        m2 = int(len(ls)/2)
        return (ls[m1]+ls[m2])/2



def nrmse(rmse, y_test):   
    nrmse = (rmse) / ((max(y_test) - min(y_test)))
    return nrmse[0]



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

csvFileName = '../data/results_GSR_SubjectBasedByVideo_NN.csv'

#read in CSV file
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'w', newline = '') as f:
        
        header = ['Fold', 'RMSE', 'NRMSE', 'LRRMSE', 'LRNRMSE', 'DTRMSE', 'DTNRMSE'] 
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

#replace NaN with 0
dataset = dataset.fillna(0)

print("Evaluating...")
        

X = dataset.iloc[:, df.columns != 'GSR_mean']
y = dataset.iloc[:, df.columns == 'GSR_mean'].values

print("X is :", X.shape)
print("y is :", y.shape)


print("Reading data for preprocessing...")
path = 'dataset path' # use your path # use your path
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


full_list_video = []
first_part_video = []
second_part_video = []
third_part_video = []
fourth_part_video = []
fifth_part_video = []


for normloop in range(1, 154): #154
    
    participant = normloop
    
  
    
    flag = False
    fileReader = []
    participants_video = []
    
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
            participants_video.append(filename)
            
            flag = True
    
    if flag == True:
        
        full_list.extend(fileReader)
        full_list_video.extend(participants_video)
        
        count = count + 1
        
    if (count/29) == 1:
        
        first_part.extend(full_list)
        first_part_video.extend(full_list_video)
        
        full_list = []
        full_list_video = []
        
    elif (count/29) == 2:
        
        second_part.extend(full_list)
        second_part_video.extend(full_list_video)
        
        full_list = []
        full_list_video = []
        
    elif (count/29) == 3:
        
        third_part.extend(full_list)
        third_part_video.extend(full_list_video)
        
        full_list = []
        full_list_video = []
        
    elif (count/29) == 4:
        
        fourth_part.extend(full_list)
        fourth_part_video.extend(full_list_video)
        
        full_list = []
        full_list_video = []
        
    elif (count/29) == 5:
        
        fifth_part.extend(full_list)
        fifth_part_video.extend(full_list_video)
        
        full_list = []
        full_list_video = []

list_of_lists = [first_part, second_part, third_part, fourth_part, fifth_part]

for i in range(len(list_of_lists)):
    
    print("***********Splitting test and train...")
    train_lists = list_of_lists[0:i] + list_of_lists[i+1:]
    test_list = list_of_lists[i]
 
    
    print("***********Splitting train and eval...")
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
    
   
    #dropna 
    train_data = train_data.dropna()
      
    X_train = train_data.iloc[:, train_data.columns != 'GSR_mean']
    y_train = train_data.iloc[:, train_data.columns == 'GSR_mean'].values
    
    val_data = pd.concat(val_list, axis=0, ignore_index=True)
            
    #drop NaN 
    val_data = val_data.dropna()
  
    X_val = val_data.iloc[:, val_data.columns != 'GSR_mean']
    y_val = val_data.iloc[:, val_data.columns == 'GSR_mean'].values
    
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression
        
    LinearReg = LinearRegression().fit(X_train, y_train)
    DTReg = DecisionTreeRegressor(random_state=1).fit(X_train, y_train)
    
    print("X_train :", X_train.shape)
    X_train = np.array(X_train)
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
       
    print("After reshaping...")
    print(type(X_train))
    print("X_train :", X_train.shape)
    
   
    print("X_val :", X_val.shape)
    X_val = np.array(X_val)
    X_val = np.reshape(X_val, [X_val.shape[0], X_val.shape[1], 1])
       
    print(type(X_val))
    print("X_val :", X_val.shape)  

    
    print("Evaluating "+ str(i) +" fold...")
    
    dropouts = 0.2
    regressor = model(X_train, dropouts)
    regressor.fit(X_train, y_train, epochs = 1, verbose = 1, batch_size = 64, validation_data = (X_val, y_val))
    
    
    if i == 0:
        video_list = first_part_video    
    elif i == 1:
        video_list = second_part_video
    elif i == 2:
        video_list = third_part_video
    elif i == 3:
        video_list = fourth_part_video
    elif i ==4:
        video_list = fifth_part_video
       
    for video_count in range(0, len(video_list)):
        
        test_file = pd.read_csv(video_list[video_count], encoding = 'ISO-8859-1', header = 0)
        
        test_data = test_file.dropna()
   
        X_test = test_data.iloc[:, test_data.columns != 'GSR_mean']
        y_test = test_data.iloc[:, test_data.columns == 'GSR_mean'].values
        
        LinearPred = LinearReg.predict(X_test)
        DTPred = DTReg.predict(X_test)
        
        Linear_mse = mean_squared_error(y_test, LinearPred)
        Linear_rmse = sqrt(Linear_mse)
        
        DT_mse = mean_squared_error(y_test, DTPred)
        DT_rmse = sqrt(DT_mse)
        
        print("****************************************************")
        print("Results for fold no. " + str(i))
        
        #Evaluate Performance
        print("LR Mean squared error", Linear_mse)
        print("LR Root mean squared error", Linear_rmse)
        
        NLinear_nrmse = nrmse(Linear_rmse, y_test)
        print("LR Normalized root mean squared error", NLinear_nrmse)
        
        print("****************************************************")
        print("Results for fold no. " + str(i))
        
        #Evaluate Performance
        print("DT Mean squared error", DT_mse)
        print("DT Root mean squared error", DT_rmse)
        
        NDT_nrmse = nrmse(DT_rmse, y_test)
        print("DT Normalized root mean squared error", NDT_nrmse)
        
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1], 1])
        
        # make predictions
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = sqrt(mse)
        
        print("****************************************************")
        print("Results for fold no. " + str(i))
        
        #Evaluate Performance
        print("Mean squared error", mse)
        print("Root mean squared error", rmse)
        
        nrmse_val = nrmse(rmse, y_test)
        print("Normalized root mean squared error", nrmse_val)
        
        #append and save results
        row_contents = [str(i), str(video_count), str(rmse), str(nrmse_val), str(Linear_rmse), str(NLinear_nrmse), str(DT_rmse), str(NDT_nrmse)]
        append_list_as_row(csvFileName, row_contents)
    
    
    
    
    
    
    
    
    
    
    
    