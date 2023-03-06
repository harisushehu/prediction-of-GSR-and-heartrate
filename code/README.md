Due to copyright reasons, the data is not made available. However, researchers who wish to use the Emoutional Arousal Pattern (EMAP) dataset can request to use the dataset from https://www.wgtn.ac.nz/psyc/research/emap-open-database

To run the code, put the EMAP dataset in the 'data' directory and change the dataset path such that the 'path' in all scripts will point to the location of the EMAP dataset.

Code description

run Feature_extraction.py to extract EEG (Alpha, Beta, Gamma, and Theta channel) features from a pre-processed EEG file.

run correlation.py to get the correlation between EEG, arousal ratings, and peripheral measures (i.e. heart rate, skin conductance, blood volume, and respiration).

run HeartRate.py and GSR.py to get the results of predicting heart rate and skin conductance, respectively using LSTM model, which has a temporal mechanism.

run HR_random.py, GSR_random.py, and LR_random.py to compare the results of predicting GSR and heart rate using LSTM and Linear regression with a random guess, where all labels are randomly shuffled. 

run HeartRate_FS_DT.py and HeartRate_FS_LR.py to obtain the minimum number of features that contribute to predicting heart rate using Decision Tree and Linear regression, respectively. 

run GSR_FS_DT.py and GSR_FS_LR.py to obtain the minimum number of features that contribute to predicting GSR using Decision Tree and Linear regression, respectively. 

run SelectedFeatures_GSR.py to obtain the prediction accuracy (error rate) of the LSTM model with the features selected using the feature selection technique.

All code on invesigating individual differences using the data selection approach can be found in the folder named Further_analysis