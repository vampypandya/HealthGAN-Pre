import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from data_utils import getHeader, findCorrelation, generateReport
from sklearn.metrics import mean_absolute_error
from distance_calculator import calculateDistance
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="3"

distance_parameter = ['Frobenius Norm', 'Energy Distance', 'L2 Norm', 'Wasserstein Distance', 'Frechet Inception Distance',
                      'Students T-test', 'KS Test', 'Shapiro Wil Test', 'Anderson Darling Test']


rest = ['108', '210', '157', '159']
d1 = '50'
d2 = rest
threshold = 0.3
real_train_file = 'x_train.npy'
syn_train_file = '0.npy'
header_file = "x_headers.txt"
lst = getHeader(header_file)
# Real Data and Synthetic Data
# corr_real, corr_syn, chng_lst = findCorrelation(real_train_file, syn_train_file, lst)
#
# # Deleted columns list
# del_list = [x for x in lst if x not in chng_lst]
#
# # Plot and Mean Absolute Error
# error = mean_absolute_error(corr_real, corr_syn)
# print("Mean Absolute Error: ",error)


lst = getHeader("x_headers.txt")
real_values, syn_values, real_values_rest, syn_values_rest, pairs, pairs_rest = findCorrelation(d1, d2, real_train_file, syn_train_file, lst, first = 'Real')
# Deleted columns list
# del_list = [x for x in lst if x not in chng_lst]

# Plot and Mean Absolute Error
error = mean_absolute_error(real_values, syn_values)
print("Mean Absolute Error: ",error)
error = mean_absolute_error(real_values_rest, syn_values_rest)
print("Mean Absolute Error: ",error)


## Plot 1
# data to plot
n_groups = 20
x = list(real_values)
y = list(syn_values)
# create plot
fig, ax = plt.subplots()
plt.figure(figsize=(20, 20))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, x, bar_width,
alpha=opacity,
color='b',
label='Diabetes: 50')

rects2 = plt.bar(index + bar_width, y, bar_width,
alpha=opacity,
color='g',
label='Synthetic Data')

labels = [str(pair[0])+'\n'+str(pair[1]) for pair in pairs]
plt.ylabel('Correlation Score')

plt.title('Top 20 Correlation Comparison(50 to Syn Data)')
plt.xticks(index + bar_width, labels, rotation='vertical')
plt.legend()

plt.tight_layout()
plt.show()

## Plot 2
# data to plot
n_groups = 20
x = list(real_values_rest)
y = list(syn_values_rest)
# create plot
fig, ax = plt.subplots()
plt.figure(figsize=(20, 20))
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
rects1 = plt.bar(index, x, bar_width,
alpha=opacity,
color='b',
label='Rest Data')

rects2 = plt.bar(index + bar_width, y, bar_width,
alpha=opacity,
color='g',
label='Synthetic Data')

labels = [str(pair[0])+'\n'+str(pair[1]) for pair in pairs_rest]
plt.ylabel('Correlation Score')

plt.title('Top 20 Correlation Comparison(Rest Data to Syn Data)')
plt.xticks(index + bar_width, labels, rotation='vertical')
plt.legend()

plt.tight_layout()
plt.show()

# Distance Calculator
# distances = calculateDistance(corr_real, corr_syn, distance_parameter)

# Generate Report
# generateReport(distances)
