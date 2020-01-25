import numpy as np
import pandas as pd

def getData():
    # 1. From y_headers.txt find column indexes given a list of codes. The format is ccs_[NUMBER].
    code_list = ['108', '50', '210', '157', '159']
    y_header = open("y_headers.txt", "r").read().split('\n')
    y_index = [y_header.index('personal_ccs_' + val) for val in code_list]
    print(y_index)

    # 2. Find rows with at least one non-zero value in y_train.npy for those columns.
    data = np.load('y_train.npy')
    rows_num = []
    y_data_rows = []
    for row_index, row in enumerate(data):
        for col in y_index:
            if row[col] != 0 and row_index not in rows_num:
                rows_num.append(row_index)
                y_data_rows.append(row)
    # data_rows = np.asarray([x for colm_name in y_index for x in data if x[colm_name]!=0 ])
    y_data_rows = np.asarray(y_data_rows)

    # 3. Now select these rows in 'x_train.npy'.
    data = np.load('x_train.npy')
    data = data.reshape(data.shape[0], -1)
    x_data_rows = np.asarray([data[i] for i in rows_num])
    print(x_data_rows.shape)

    # 4. Use these rows for evaluation instead of the whole matrix.
    data = np.load('0.npy')
    print(data.shape)
    return x_data_rows,data

