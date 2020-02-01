import numpy as np
import pandas as pd

def getData(d1,d2):
    # 1. From y_headers.txt find column indexes given a list of codes. The format is ccs_[NUMBER].
    # code_list = [d1,d2]
    y_header = open("y_headers.txt", "r").read().split('\n')
    print(y_header)
    x_header = open("x_headers.txt", "r").read().split('\n')
    print(x_header)
    print('Hello')
    # y_index = [y_header.index('personal_ccs_' + val) for val in code_list]
    # ind1, ind2 = y_index[0], y_index[1]
    ind1 = y_header.index('personal_ccs_' + d1)
    ind2 = [y_header.index('personal_ccs_' + val) for val in d2]

    # 2. Find rows with at least one non-zero value in y_train.npy for those columns.
    data = np.load('y_train.npy')
    print(data.shape)
    rows_num_1 = []
    rows_num_2 = []
    y_data_rows = []
    for row_index, row in enumerate(data):
        if row[ind1] != 0:
            rows_num_1.append(row_index)
        for ind in ind2:
            if row[ind] != 0 and ind not in rows_num_2:
                rows_num_2.append(row_index)

    # 3. Now select these rows in 'x_train.npy'.
    data = np.load('x_train.npy')
    data = data.reshape(data.shape[0], -1)
    print (data.shape)
    x_data_rows_1 = np.asarray([data[i] for i in rows_num_1])
    x_data_rows_2 = np.asarray([data[i] for i in rows_num_2])
    # print(x_data_rows.shape)

    # 4. Use these rows for evaluation instead of the whole matrix.
    data = np.load('0.npy')
    # print(data.shape)
    return x_data_rows_1, x_data_rows_2, data

