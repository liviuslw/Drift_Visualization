# -*- coding: utf-8 -*-
"""
 pre-processing on sea data set
"""

import pandas as pd
import numpy as np
from numpy.random import randint
from pandas import DataFrame
import os


def load_data(trainpath,testpath):
    train = pd.read_csv(trainpath, sep=',', header=None).values
    xtrain = train[:, :-1]
    # y_tmp = train[:, -1].reshape((-1,1))
    # ytrain = np.concatenate((y_tmp,1-y_tmp),axis=1)
    ytrain = train[:, -1].astype(np.int64)
    test = pd.read_csv(testpath, sep=',', header=None).values
    xtest = test[:, :-1]
    # y_tmp = test[:, -1].reshape((-1,1))
    # ytest = np.concatenate((y_tmp,1-y_tmp),axis=1)
    ytest = test[:, -1].astype(np.int64)

    out = (xtrain, ytrain, xtest, ytest)
    return out


def load_data_withclass(trainpath,train_label_path,testpath,test_label_path):
    xtrain = pd.read_csv(trainpath, sep=',', header=None).values
    # y_tmp = train[:, -1].reshape((-1,1))
    # ytrain = np.concatenate((y_tmp,1-y_tmp),axis=1)
    ytrain = np.squeeze(pd.read_csv(train_label_path, sep=',', header=None).values-1).astype(np.int64)
    xtest = pd.read_csv(testpath, sep=',', header=None).values
    # y_tmp = test[:, -1].reshape((-1,1))
    # ytest = np.concatenate((y_tmp,1-y_tmp),axis=1)
    ytest = np.squeeze(pd.read_csv(test_label_path, sep=',', header=None).values-1).astype(np.int64)
    out = (xtrain, ytrain, xtest, ytest)
    return out


def data_augmentation(data,batchsize,duplicate_rate):
    xtrain, ytrain, xtest, ytest = data
    batch = xtrain.shape[0]/batchsize
    duplic_batchsize = batchsize * duplicate_rate
    # initialization
    xtrain_aug = np.zeros((xtrain.shape[0]*duplicate_rate, xtrain.shape[1]))
    ytrain_aug = np.zeros((ytrain.shape[0]*duplicate_rate))
    xtest_aug = np.zeros((xtest.shape[0]*duplicate_rate, xtest.shape[1]))
    ytest_aug = np.zeros((ytest.shape[0]*duplicate_rate))


    # replicate data within every batch

    rand_ind = []
    # generate index sequence for many times
    for rand in range(duplicate_rate):
        rand_ind.append(randint(0, batchsize, batchsize))

    for i in range(batch):
        temp_xtrain = xtrain[i * batchsize:(i + 1) * batchsize, :]
        temp_ytrain = ytrain[i * batchsize:(i + 1) * batchsize]
        temp_xtest = xtest[i * batchsize:(i + 1) * batchsize, :]
        temp_ytest = ytest[i * batchsize:(i + 1) * batchsize]

        for j in range(duplicate_rate):
            xtrain_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1)),:]\
                = temp_xtrain[rand_ind[j],:]
            ytrain_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1))]\
                = temp_ytrain[rand_ind[j]]
            xtest_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1)),:]\
                = temp_xtest[rand_ind[j], :]
            ytest_aug[(i*duplic_batchsize + batchsize*j):(i*duplic_batchsize+batchsize*(j+1))] \
                = temp_ytest[rand_ind[j]]

    out = (xtrain_aug,ytrain_aug,xtest_aug,ytest_aug)
    return out


def batch_split(data,train_split_size, test_split_size):
    xtrain, ytrain, xtest, ytest = data
    batch = xtrain.shape[0]//train_split_size
    batchdata = []
    for i in range(batch):
        train_index = range(i*train_split_size,(i+1)*train_split_size)
        batchtrain_x = xtrain[train_index,:]
        batchtrain_y = ytrain[train_index]
        test_index = range(i*test_split_size,(i+1)*test_split_size)
        batchtest_x = xtest[test_index,:]
        batchtest_y = ytest[test_index]
        batchdata.append({"xtrain":batchtrain_x,
                          "ytrain":batchtrain_y,
                          "xtest":batchtest_x,
                          "ytest":batchtest_y
        })
    return batchdata


def data_preprocessing(path, train_batchsize = 0, test_batchsize = 0):
    data_filename = os.path.split(path)
    if data_filename[-1] == 'sea':
        trainpath = path + '/seadata.csv'
        testpath = path + '/seatest.csv'
        data = load_data(trainpath, testpath)
    elif data_filename[-1] == 'checkerboard_data':
        trainpath = path + '/CBconstant_testing_data.csv'
        train_label_path = path + '/CBconstant_testing_class.csv'
        testpath = path + '/CBconstant_testing_data.csv'
        test_label_path = path + '/CBconstant_testing_class.csv'
        data = load_data_withclass(trainpath,train_label_path,testpath,test_label_path)
    duplicate_rate = 5
    if train_batchsize or test_batchsize:
    # data = data_augmentation(data, batchsize=batchsize, duplicate_rate=duplicate_rate)
        batchdata = batch_split(data, train_split_size=train_batchsize,test_split_size=test_batchsize)
    else:
        batchdata = {}
        batchdata["xtrain"] = data[0]
        batchdata["ytrain"] = data[1]
        batchdata["xtest"] = data[2]
        batchdata["ytest"] = data[3]
    return batchdata


def save_acc(acc):
    pd = DataFrame(acc)
    pd.to_csv('heap_map.csv',header=False,index=False)
