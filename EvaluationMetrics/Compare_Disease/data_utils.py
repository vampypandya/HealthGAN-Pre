import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
import os
from data_gen import getData

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# def getData(file_name, both=False):
#     data = np.load(file_name)
#     data = data.reshape(data.shape[0], -1)
#     return data


def getHeader(file_name):
    lst_x = open(file_name, "r").read().split('\n')
    lst = [x + '_1' for x in lst_x] + [x + '_2' for x in lst_x]
    return lst


def getDataframes(data, lst):
    data1 = data[[x for x in lst if x[0] == 'i']]
    data2 = data[[x for x in lst if x[0] == 'a']]
    return data1, data2


def thresholdCorr(corr, diag, meds, maxAmt, c_index):
    corr_new = sorted(list(corr), key=abs, reverse=True)[:maxAmt]
    index_max = [list(corr).index(i) for i in corr_new]
    pair = []
    for each_index in index_max:
        coords = c_index[each_index]
        new_pair = [diag[coords[0]], meds[coords[1]], corr[each_index]]
        pair.append(new_pair)
    return corr_new, index_max, pair


def getCorrMatrix(diag, meds):
    corr = []
    corr_indexes = []
    diag = diag.astype(float)
    meds = meds.astype(float)
    print(diag.shape)
    print(meds.shape)
    for i, diagnosis in tqdm(enumerate(diag)):
        for j, medication in enumerate(meds):
            corr_indexes.append([i, j])
            diagnosis_t = tf.convert_to_tensor(diagnosis)
            medication_t = tf.convert_to_tensor(medication)
            corr_val = tfp.stats.correlation(diagnosis_t, medication_t, sample_axis=0, event_axis=None)
            corr.append(corr_val)
    return corr, corr_indexes


def removeZero(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    return df, list(df.columns)


def findCorrelation(d1, d2, real_train_file, syn_train_file, lst, first, threshold=0.5):
    data1, data2 , data_syn = getData(d1, d2)
    # data_syn = getData(syn_train_file)
    main_lst = getHeader("x_headers.txt")
    # Update Real Training DataFrame and remove Zero columns

    df1 = pd.DataFrame(data1, columns=main_lst)
    df1 = df1.drop(['atc3_ANTIINFECTIVES_1', 'atc3_ANTIINFECTIVES_2'], axis=1)
    df1, lst = removeZero(df1)

    df2 = pd.DataFrame(data2, columns=main_lst)
    df2 = df2.drop(['atc3_ANTIINFECTIVES_1', 'atc3_ANTIINFECTIVES_2'], axis=1)

    # Update Synthetic Training DataFrame and remove Zero columns
    df_syn = pd.DataFrame(data_syn, columns=main_lst)
    df_syn = df_syn[lst]
    df_syn, lst = removeZero(df_syn)

    # Again update real training data to match the column names with Synthetic data
    df1 = df1[lst]
    df2 = df2[lst]



    #Check for Single to Syn
    # if first =='Real':
    # Find real data dataframs of meds and diag and find correlation between them
    diag, meds = getDataframes(df1, lst)
    # TODO Temporary arrangement
    # corr, c_index = getCorrMatrix(diag.T.values, meds.T.values)
    # corr = np.asarray(corr)
    # np.save('single_data_R2S', corr)
    # c_index = np.asarray(c_index)
    # np.save('single_data_R2S_index', c_index)
    ######
    corr = np.load('single_data_R2S.npy')
    c_index = np.load('single_data_R2S_index.npy')
    real_values, indexList, pairs = thresholdCorr(corr, diag.columns, meds.columns, 20, c_index)
    # Find synthetic data dataframs of meds and diag and find correlation between them
    diag_syn, meds_syn = getDataframes(df_syn, lst)
    # TODO Temporary arrangement
    # corr_syn, csyn_index = getCorrMatrix(diag_syn.T.values, meds_syn.T.values)
    # corr_syn = np.asarray(corr_syn)
    # np.save('single_data_R2S_2', corr_syn)
    #######
    corr_syn = np.load('single_data_R2S_2.npy')
    corr_syn = list(corr_syn)
    syn_values = [corr_syn[x] for x in indexList]


    ## Collect Data for single
    # After getting the data
    d_cols = diag.columns
    m_cols = meds.columns
    pair = []
    for each_index in indexList:
        coords = c_index[each_index]
        new_pair = [d_cols[coords[0]], m_cols[coords[1]], corr[each_index], corr_syn[each_index]]
        pair.append(new_pair)

    ######################


    # Check for Rest to Syn
    # if first =='Real':
    # Find real data dataframs of meds and diag and find correlation between them
    diag, meds = getDataframes(df2, lst)
    # TODO Temporary arrangement
    # corr, c_index = getCorrMatrix(diag.T.values, meds.T.values)
    # corr = np.asarray(corr)
    # np.save('rest_data_R2S', corr)
    # c_index = np.asarray(c_index)
    # np.save('rest_data_R2S_index', c_index)
    ######
    corr = np.load('rest_data_R2S.npy')
    c_index = np.load('rest_data_R2S_index.npy')
    real_values_rest, indexList, pairs = thresholdCorr(corr, diag.columns, meds.columns, 20, c_index)
    # Find synthetic data dataframs of meds and diag and find correlation between them
    diag_syn, meds_syn = getDataframes(df_syn, lst)
    # TODO Temporary arrangement
    # corr_syn, csyn_index = getCorrMatrix(diag_syn.T.values, meds_syn.T.values)
    # corr_syn = np.asarray(corr_syn)
    # np.save('rest_data_R2S_2', corr_syn)
    #######
    corr_syn = np.load('rest_data_R2S_2.npy')
    corr_syn = list(corr_syn)
    syn_values_rest = [corr_syn[x] for x in indexList]

    ## Collect Data for rest
    # After getting the data
    d_cols = diag.columns
    m_cols = meds.columns
    pair_rest = []
    for each_index in indexList:
        coords = c_index[each_index]
        new_pair = [d_cols[coords[0]], m_cols[coords[1]], corr[each_index], corr_syn[each_index]]
        pair_rest.append(new_pair)
    #####################
    df_pairs = pd.DataFrame(pair)
    df_pairs.to_csv('main_data.csv', sep=',', header=None, index=None)
    return real_values, syn_values, real_values_rest, syn_values_rest, pair, pair_rest


def generateReport(distances):
    # TODO Generate proper result
    print(distances)
