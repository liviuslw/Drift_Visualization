# -*- coding: utf-8 -*-
"""
 Algorithm on sea data set
"""

import numpy as np
import preprocessing_for_sea as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pandas import DataFrame
import seaborn

def visualize_out_prob(out_prob,batchdata,decision_bound):
    animation_with_label(out_prob,batchdata,decision_bound)


def animation_with_label(out_prob,batchdata,decision_bound):


    # build plot
    fig, axes = plt.subplots(1,2)
    ax,ax2 = axes.ravel()


    # build static plot
    x = out_prob[0]
    y = batchdata[0]["ytest"]


    # plot points
    cm_dark = mpl.colors.ListedColormap(['navy', 'orange'])
    sca = ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=cm_dark)
    ax.set_title('time step {0}'.format(0))
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')

    # build dynamic plot
    def update(i):
        # label = 'timestep {0}'.format(i)
        # print(label)
        x = out_prob[i]
        y = batchdata[i]["ytest"]
        data = [[x1,x2] for x1, x2 in zip(x[:,0],x[:,1])]
        sca.set_offsets(data)
        sca.set_array(y.ravel())
        # cm_dark = mpl.colors.ListedColormap(['r', 'g'])
        # plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=cm_dark)
        ax.set_title('time step {0}'.format(i))
        return sca
    anim = FuncAnimation(fig, update,frames= range(len(batchdata)),interval=100)


    # seaborn.set_style("whitegrid")
    # build plot

    # build static plot
    x2 = batchdata[0]["xtrain"]
    y2 = batchdata[0]["ytrain"]
    cm_dark = mpl.colors.ListedColormap(['navy', 'orange'])
    sca2 = ax2.scatter(x2[:, 0], x2[:, 1], c=y2.ravel(), s=40, cmap=cm_dark)
    ax2.set_title('time step {0}'.format(0))
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    # plot decision boundary
    xx,yy,Z = decision_bound
    Z = Z.reshape(xx.shape)
    # ax2.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)

    # build dynamic plot
    def update2(i):
        # label = 'timestep {0}'.format(i)
        # print(label)
        x2 = batchdata[i]["xtrain"]
        y2 = batchdata[i]["ytrain"]
        data2 = [[x1,x3] for x1, x3 in zip(x2[:,0],x2[:,1])]
        sca2.set_offsets(data2)
        sca2.set_array(y2.ravel())
        # cm_dark = mpl.colors.ListedColormap(['r', 'g'])
        # plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=cm_dark)
        ax2.set_title('time step {0}'.format(i))
        return sca2
    anim2 = FuncAnimation(fig, update2,frames= range(len(batchdata)),interval=100)


    plt.show()


def animation(batchdata):
    # seaborn.set_style("whitegrid")
    # build plot
    fig, ax = plt.subplots()

    # build static plot
    x = batchdata[0]["xtrain"]
    y = batchdata[0]["ytrain"]
    cm_dark = mpl.colors.ListedColormap(['r', 'g'])
    sca = plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=15, cmap=cm_dark)
    plt.title('time step {0}'.format(0))
    # build dynamic plot
    def update(i):
        # label = 'timestep {0}'.format(i)
        # print(label)
        x = batchdata[i]["xtrain"]
        y = batchdata[i]["ytrain"]
        data = [[x1,x2] for x1, x2 in zip(x[:,0],x[:,1])]
        sca.set_offsets(data)
        sca.set_array(y.ravel())
        # cm_dark = mpl.colors.ListedColormap(['r', 'g'])
        # plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=cm_dark)
        plt.title('time step {0}'.format(i))
        return sca
    anim = FuncAnimation(fig, update,frames= range(len(batchdata)),interval=200)
    plt.show()


def main():
    path = 'data/sea/'
    batchdata = pd.data_preprocessing(path,split=True)

    animation(batchdata)


if __name__ == '__main__':
    main()
