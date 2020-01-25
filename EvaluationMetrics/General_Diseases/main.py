import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from data_utils import getHeader, findCorrelation, generateReport
from sklearn.metrics import mean_absolute_error
from distance_calculator import calculateDistance
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"

distance_parameter = ['Frobenius Norm', 'Energy Distance', 'L2 Norm', 'Wasserstein Distance', 'Frechet Inception Distance',
                      'Students T-test', 'KS Test', 'Shapiro Wil Test', 'Anderson Darling Test']

threshold = 0.3
real_train_file = 'x_train.npy'
syn_train_file = '0.npy'
header_file = "x_headers.txt"
lst = getHeader(header_file)
# Real Data and Synthetic Data
corr_real, corr_syn, chng_lst = findCorrelation(real_train_file, syn_train_file, lst)

# Deleted columns list
del_list = [x for x in lst if x not in chng_lst]

# Plot and Mean Absolute Error
error = mean_absolute_error(corr_real, corr_syn)
print("Mean Absolute Error: ",error)

x = np.concatenate((corr_real, corr_syn))
y = np.random.rand(x.shape[0])
labels = ['Real Data']*100 + ['Syn Data']*100
df = pd.DataFrame(dict(x=x, y=y, label=labels))
groups = df.groupby('label')
fig, ax = plt.subplots()
ax.margins(0.05)
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=name)
ax.legend()

plt.show()
plt.savefig('data_merror.png')
# Distance Calculator
# distances = calculateDistance(corr_real, corr_syn, distance_parameter)

# Generate Report
# generateReport(distances)
