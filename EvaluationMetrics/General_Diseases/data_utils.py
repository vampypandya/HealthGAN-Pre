import heapq
import operator
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def getData(file_name, both = False):
    data = np.load(file_name)
    data = data.reshape(data.shape[0], -1)
    return data

def getHeader(file_name):
    lst_x = open(file_name, "r").read().split('\n')
    lst = [x+'_1' for x in lst_x] + [x+'_2' for x in lst_x]
    return lst

def getDataframes(data,lst):
    data1 = data[[x for x in lst if x[0] == 'i']]
    data2 = data[[x for x in lst if x[0] == 'a']]
    return data1, data2


def thresholdCorr(corr, diag, meds, maxAmt):
    print(type(corr),len(diag)*len(meds))
    index_max = list(zip(*heapq.nlargest(maxAmt, enumerate(list(corr)), key=operator.itemgetter(1))))[0]
    # index_min = list(zip(*heapq.nsmallest(maxAmt, enumerate(list(corr)), key=operator.itemgetter(1))))[0]
    real_vals = [corr[x] for x in index_max]
    return real_vals, index_max

def getCorrMatrix(diag, meds):
    corr = []
    diag = diag.astype(float)
    meds = meds.astype(float)
    # corr_val = tfp.stats.correlation(diag_t, med_t)
    for diagnosis in tqdm(diag):
        for medication in meds:
            diagnosis_t = tf.convert_to_tensor(diagnosis)
            medication_t = tf.convert_to_tensor(medication)
            corr_val = tfp.stats.correlation(diagnosis_t, medication_t, sample_axis=0, event_axis=None)
            # corr_val = np.corrcoef([diagnosis, medication])
            corr.append(corr_val)

    return corr



def findCorrelation(real_train_file, syn_train_file, lst, threshold = 0.5):
    data = getData(real_train_file)
    data_syn = getData(syn_train_file)
    main_lst = lst
    # Update Real Training DataFrame and remove Zero columns

    df = pd.DataFrame(data, columns=main_lst)
    df = df.drop(['atc3_ANTIINFECTIVES_1', 'atc3_ANTIINFECTIVES_2'],axis = 1)
    df, lst = removeZero(df)

    # Update Synthetic Training DataFrame and remove Zero columns
    df_syn = pd.DataFrame(data_syn, columns=main_lst)
    df_syn = df_syn[lst]
    df_syn, lst = removeZero(df_syn)

    # Again update real training data to match the column names with Synthetic data
    df = df[lst]
    # Find real data dataframs of meds and diag and find correlation between them
    diag, meds = getDataframes(df, lst)

    # TODO Temporary arrangement
    # corr = getCorrMatrix(diag.T.values, meds.T.values)
    # corr = np.asarray(corr)
    # np.save('data', corr)
    ######


    # np.fill_diagonal(corr, np.nan)
    corr = np.load('data.npy')
    real_values, indexList = thresholdCorr(corr, diag.columns, meds.columns, 100)


    # Find synthetic data dataframs of meds and diag and find correlation between them
    diag_syn, meds_syn = getDataframes(df_syn, lst)
    # TODO Temporary arrangement
    # corr_syn = getCorrMatrix(diag_syn.T.values, meds_syn.T.values)
    # corr_syn = np.asarray(corr_syn)
    # np.save('data_syn', corr_syn)
    #######
    corr_syn = np.load('data_syn.npy')
    corr_syn = list(corr_syn)
    syn_values = [corr_syn[x] for x in indexList]
    return real_values, syn_values, lst

def removeZero(df):
    df = df.loc[:, (df != 0).any(axis=0)]
    return df, list(df.columns)

def generateReport(distances):
    # TODO Generate proper result
    print(distances)