# -*- coding: utf-8 -*-
"""
 data processing
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.random import randint
import json
from scipy.io import arff


def load_data(datapath, labelpath):
    x = pd.read_csv(datapath, sep=',| ', header=None,engine='python').values
    y = np.squeeze(pd.read_csv(labelpath, sep=' ', header=None).values)
    out = (x, y)
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
    x, y = data
    batch = x.shape[0]//batchsize
    batchdata = []
    for i in range(batch-1):
        index = range(i*batchsize,(i+1)*batchsize)
        batchtrain_x = x[index,:]
        batchtrain_y = y[index]
        batchtest_x = x[np.array(index).reshape(-1)+batchsize,:]
        batchtest_y = y[np.array(index).reshape(-1)+batchsize]
        batchdata.append({"xtrain":batchtrain_x,
                          "ytrain":batchtrain_y,
                          "xtest":batchtest_x,
                          "ytest":batchtest_y
        })
    return batchdata


def data_preprocessing(outpath,batchsize):
    trainpath, testpath = outpath
    data = load_data(trainpath, testpath)
    duplicate_rate = 5
    # data = data_augmentation(data, batchsize=batchsize, duplicate_rate=duplicate_rate)
    batchdata = batch_split(data, batchsize=batchsize)
    return batchdata


def dataset_config(name):
    config_path = open('Config/'+name)
    jsonData = json.load(config_path)

    filepath = 'data/' + jsonData['filepath']
    xpath = filepath + name + '.data'
    ypath = filepath + name + '.labels'

    # special cases for irregular formats
    if name == 'weather':
        xpath = filepath + 'NEweather_data.csv'
        ypath = filepath + 'NEweather_class.csv'
    if name == 'Elec2':
        xpath = filepath + 'elec2_data.dat'
        ypath = filepath + 'elec2_label.dat'
    if name == 'covType':
        xpath = filepath + 'covType.arff'
        ypath = xpath

    parameters = {
        "outpath":(xpath, ypath),
        "batchsize": jsonData['batchsize'],
        "axesrange": jsonData['axesrange'],
        "zaxesrange": jsonData['zaxesrange'],
        "num_epochs": jsonData['num_epochs'],
        "learning_rate":jsonData['learning_rate']
    }

    return parameters

