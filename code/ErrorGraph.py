# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:06:13 2021

@author: harisushehu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


fig = plt.figure()
yerr = np.linspace(0.05, 0.2, 5)
upperlimits = [True, False] * 5
lowerlimits = [False, True] * 5

#--------------------plot for GSR------------------------>

x = [1, 2, 3, 4, 5]
column_names = ["NRMSE"]

df_GSR0 = pd.read_csv("../data/Results/GSR/results_GSR_shift0.csv", usecols = column_names)
df_GSR5 = pd.read_csv("../data/Results/GSR/results_GSR_shift5.csv", usecols = column_names)
df_GSR10 = pd.read_csv("../data/Results/GSR/results_GSR_shift10.csv", usecols = column_names)
df_GSR15 = pd.read_csv("../data/Results/GSR/results_GSR_shift15.csv", usecols = column_names)
df_GSR20 = pd.read_csv("../data/Results/GSR/results_GSR_shift20.csv", usecols = column_names)
df_GSR25 = pd.read_csv("../data/Results/GSR/results_GSR_shift25.csv", usecols = column_names)


df_GSR0_std = np.std(df_GSR0["NRMSE"])
df_GSR5_std = np.std(df_GSR5["NRMSE"])
df_GSR10_std = np.std(df_GSR10["NRMSE"])
df_GSR15_std = np.std(df_GSR15["NRMSE"])
df_GSR20_std = np.std(df_GSR20["NRMSE"])
df_GSR25_std = np.std(df_GSR25["NRMSE"])

df_GSR0_mean = np.mean(df_GSR0["NRMSE"])
df_GSR5_mean = np.mean(df_GSR5["NRMSE"])
df_GSR10_mean = np.mean(df_GSR10["NRMSE"])
df_GSR15_mean = np.mean(df_GSR15["NRMSE"])
df_GSR20_mean = np.mean(df_GSR20["NRMSE"])
df_GSR25_mean = np.mean(df_GSR25["NRMSE"])



labels = ['Shift 0', 'Shift 5', 'Shift 10', 'Shift 15', 'Shift 20', 'Shift 25']
x_pos = np.arange(len(labels))
error = [df_GSR0_std, df_GSR5_std, df_GSR10_std, df_GSR15_std, df_GSR20_std, df_GSR25_std]
CTEs = [df_GSR0_mean, df_GSR5_mean, df_GSR10_mean, df_GSR15_mean, df_GSR20_mean, df_GSR25_mean]

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('NRMSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('GSR Bar Chart Showing NRMSE with different Shift')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()



plt.errorbar(x, df_GSR0["NRMSE"], yerr=yerr, uplims=True, label='Shift 0')
plt.errorbar(x, df_GSR5["NRMSE"], yerr=yerr, uplims=True, label='Shift 5')
plt.errorbar(x, df_GSR10["NRMSE"], yerr=yerr, uplims=True, label='Shift 10')
plt.errorbar(x, df_GSR15["NRMSE"], yerr=yerr, uplims=True, label='Shift 15')
plt.errorbar(x, df_GSR20["NRMSE"], yerr=yerr, uplims=True, label='Shift 20')
plt.errorbar(x, df_GSR25["NRMSE"], yerr=yerr, uplims=True, label='Shift 25')

plt.legend(loc='lower left')


#--------------------plot for Heart Rate------------------------>

x = [1, 2, 3, 4, 5]
column_names = ["NRMSE"]

df_HeartRate0 = pd.read_csv("../data/Results/HeartRate/results_HeartRate_shift0.csv", usecols = column_names)
df_HeartRate5 = pd.read_csv("../data/Results/HeartRate/results_HeartRate_shift5.csv", usecols = column_names)
df_HeartRate10 = pd.read_csv("../data/Results/HeartRate/results_HeartRate_shift10.csv", usecols = column_names)
df_HeartRate15 = pd.read_csv("../data/Results/HeartRate/results_HeartRate_shift15.csv", usecols = column_names)
df_HeartRate20 = pd.read_csv("../data/Results/HeartRate/results_HeartRate_shift20.csv", usecols = column_names)
df_HeartRate25 = pd.read_csv("../data/Results/HeartRate/results_HeartRate_shift25.csv", usecols = column_names)


df_HeartRate0_std = np.std(df_HeartRate0["NRMSE"])
df_HeartRate5_std = np.std(df_HeartRate5["NRMSE"])
df_HeartRate10_std = np.std(df_HeartRate10["NRMSE"])
df_HeartRate15_std = np.std(df_HeartRate15["NRMSE"])
df_HeartRate20_std = np.std(df_HeartRate20["NRMSE"])
df_HeartRate25_std = np.std(df_HeartRate25["NRMSE"])

df_HeartRate0_mean = np.mean(df_HeartRate0["NRMSE"])
df_HeartRate5_mean = np.mean(df_HeartRate5["NRMSE"])
df_HeartRate10_mean = np.mean(df_HeartRate10["NRMSE"])
df_HeartRate15_mean = np.mean(df_HeartRate15["NRMSE"])
df_HeartRate20_mean = np.mean(df_HeartRate20["NRMSE"])
df_HeartRate25_mean = np.mean(df_HeartRate25["NRMSE"])



labels = ['Shift 0', 'Shift 5', 'Shift 10', 'Shift 15', 'Shift 20', 'Shift 25']
x_pos = np.arange(len(labels))
error = [df_HeartRate0_std, df_HeartRate5_std, df_HeartRate10_std, df_HeartRate15_std, df_HeartRate20_std, df_HeartRate25_std]
CTEs = [df_HeartRate0_mean, df_HeartRate5_mean, df_HeartRate10_mean, df_HeartRate15_mean, df_HeartRate20_mean, df_HeartRate25_mean]

# Build the plot   #Error bars represent standard errors of the mean

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('NRMSE')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('Heart Rate Bar Chart Showing NRMSE with different Shift')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()



plt.errorbar(x, df_HeartRate0["NRMSE"], yerr=yerr, uplims=True, label='Shift 0')
plt.errorbar(x, df_HeartRate5["NRMSE"], yerr=yerr, uplims=True, label='Shift 5')
plt.errorbar(x, df_HeartRate10["NRMSE"], yerr=yerr, uplims=True, label='Shift 10')
plt.errorbar(x, df_HeartRate15["NRMSE"], yerr=yerr, uplims=True, label='Shift 15')
plt.errorbar(x, df_HeartRate20["NRMSE"], yerr=yerr, uplims=True, label='Shift 20')
plt.errorbar(x, df_HeartRate25["NRMSE"], yerr=yerr, uplims=True, label='Shift 25')

plt.legend(loc='lower left')

