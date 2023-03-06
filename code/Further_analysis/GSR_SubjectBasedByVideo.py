# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:51:02 2021

@author: harisushehu
"""

#Go to the project directory/path and run  /home/harisushehu/myenv/bin/python GSR_SubectBasedByVideo.py

import pandas as pd
import glob
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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


#Save results
import csv
from csv import writer

#Append data in csv function
def append_list_as_row(file_name, list_of_elem):
   
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

csvFileName = '../data/results_GSR_SubjectBasedByVideo.csv'

#read in CSV file
if os.path.exists(csvFileName):
    print()
else:
    with open(csvFileName, 'w', newline = '') as f:
        
        header = ['Fold', 'Video_no', 'RMSE', 'NRMSE', 'LRRMSE', 'LRNRMSE', 'DTRMSE', 'DTNRMSE'] 
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
      
    train_X = train_data.iloc[:, train_data.columns != 'GSR_mean']
    train_y = train_data.iloc[:, train_data.columns == 'GSR_mean'].values
    
    
    #scale train                
    scaler_X1 = StandardScaler()
    scaler_y1 = StandardScaler()
        
    #normalize train X and y
    X_train = scaler_X1.fit_transform(train_X)
    y_train = scaler_y1.fit_transform(train_y)
    
    val_data = pd.concat(val_list, axis=0, ignore_index=True)
            
    #drop NaN 
    val_data = val_data.dropna()
  
    val_X = val_data.iloc[:, val_data.columns != 'GSR_mean']
    val_y = val_data.iloc[:, val_data.columns == 'GSR_mean'].values
    
    #scale val              
    scaler_X3 = StandardScaler()
    scaler_y3 = StandardScaler()
    
    #normalize test X and y
    X_val = scaler_X3.fit_transform(val_X)
    y_val = scaler_y3.fit_transform(val_y) 
    
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
    
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    keras_callbacks   = [
          EarlyStopping(monitor='val_loss',patience=3, min_delta=1e-3,mode='min'),
          ModelCheckpoint("best_GSR_Participant_model.hdf5", monitor='val_loss', save_best_only=True, mode='min', save_freq='epoch')]
    
    
    dropouts = 0.2
    regressor = model(X_train, dropouts)
    
    history = regressor.fit(X_train, y_train, epochs = 100, verbose = 1, callbacks=keras_callbacks, batch_size = 256, validation_data = (X_val, y_val))
    
    from tensorflow.keras.models import load_model
    # load best model from single file
    regressor = load_model('best_GSR_Participant_model.hdf5')
    
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
   
        test_X = test_data.iloc[:, test_data.columns != 'GSR_mean']
        test_y = test_data.iloc[:, test_data.columns == 'GSR_mean'].values
    
        #scale test                
        scaler_X2 = StandardScaler()
        scaler_y2 = StandardScaler()
        
        #normalize test X and y
        X_test = scaler_X2.fit_transform(test_X)
        y_test = scaler_y2.fit_transform(test_y)
        
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
    
        predictions = scaler_y2.inverse_transform(y_pred)
            
        valid = test_data
        #predicted = np.array(predictions)
        all_predictions = predictions     
        valid['Predictions'] = all_predictions
        
        from matplotlib import pyplot as plt 
        plt.clf()
        f, ax = plt.subplots(1)
        #plt.title('Model')
        plt.xlabel('Time', fontsize=10)
        plt.ylabel('GSR', fontsize=10)
        #plt.plot(train['GSR_mean'])
        plt.plot(valid[['GSR_mean', 'Predictions']])
        plt.legend(['Test', 'Predictions'], loc='lower right')
        ax.set_ylim(ymin=-3e-5, ymax = 5e-5)
        ax.set_xlim(xmin=0, xmax = 200)
        #plt.show()
        
        filename = "../data/GSR_SubjectByVideo/Pred_" + str(i) + "_" + str(video_count)
        plt.savefig(filename)
    
    
    
    
    
    
    
    
    
    
    
    