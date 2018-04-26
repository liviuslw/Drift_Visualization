# -*- coding: utf-8 -*-
"""
 pre-processing on sea data set
"""

import pandas as pd
import numpy as np
from numpy.random import randint
from pandas import DataFrame


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


def batch_split(data,batchsize):
    xtrain, ytrain, xtest, ytest = data
    batch = xtrain.shape[0]//batchsize
    batchdata = []
    for i in range(batch):
        index = range(i*batchsize,(i+1)*batchsize)
        batchtrain_x = xtrain[index,:]
        batchtrain_y = ytrain[index]
        batchtest_x = xtest[index,:]
        batchtest_y = ytest[index]
        batchdata.append({"xtrain":batchtrain_x,
                          "ytrain":batchtrain_y,
                          "xtest":batchtest_x,
                          "ytest":batchtest_y
        })
    return batchdata


def data_preprocessing(path,split = True):
    trainpath = path + 'seadata.csv'
    testpath = path + 'seatest.csv'
    data = load_data(trainpath, testpath)
    duplicate_rate = 5
    if split:
        batchsize = 250
    # data = data_augmentation(data, batchsize=batchsize, duplicate_rate=duplicate_rate)
        batchdata = batch_split(data, batchsize=batchsize)
    else:
        batchdata = {}
        batchdata["xtrain"] = data[0]
        batchdata["ytrain"] = data[1]
        batchdata["xtest"] = data[2]
        batchdata["ytest"] = data[3]
    return batchdata


def save_acc(acc):
    pd = DataFrame(acc)
    pd.to_csv('Result/sea/MDDT_broadscale.csv',header=False,index=False)
