# -*- coding: utf-8 -*-
"""
 Algorithm on sea data set
"""

import numpy as np
import preprocessing_for_sea as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from pandas import DataFrame
# import seaborn


def getClassColors():
    """
    Returns various different colors.
    """
    return np.array(['#000080', '#ACE600', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC',
                     '#915200', '#D9007E', '#FFCCCC', '#5E6600', '#FFFF00', '#999999',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED', '#915200',
                     '#0000FF', '#FF0000', '#00CC01', '#2F2F2F', '#8900CC', '#0099CC',
                     '#ACE600', '#D9007E', '#FFCCCC', '#5E6600', '#FFFF00', '#999999',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED', '#FFA500'])


def visualize_out_prob(out_prob,batchdata,decision_bound,parameters=None):
    animation_with_label(out_prob,batchdata,decision_bound,parameters)


def animation_with_label(out_prob,batchdata,decision_bound,parameters=None):


    # build plot
    fig, axes = plt.subplots(1,2)
    ax,ax2 = axes.ravel()


    # build static plot
    x = out_prob[0]
    y = batchdata[0]["ytest"]


    # plot points
    cm_dark = mpl.colors.ListedColormap(getClassColors())
    sca = ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=15, cmap=cm_dark)
    ax.set_title('time step {0}'.format(0))
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    # if parameters:
    #     xmin = parameters["zaxesrange"]["xmin"]
    #     xmax = parameters["zaxesrange"]["xmax"]
    #     ymin = parameters["zaxesrange"]["ymin"]
    #     ymax = parameters["zaxesrange"]["ymax"]
    #     ax.set_xlim(xmin,xmax)
    #     ax.set_ylim(ymin,ymax)

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
    cm_dark = mpl.colors.ListedColormap(getClassColors())
    sca2 = ax2.scatter(x2[:, 0], x2[:, 1], c=y2.ravel(), s=15, cmap=cm_dark)
    ax2.set_title('time step {0}'.format(0))
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    if parameters:
        xmin = parameters["axesrange"]["xmin"]
        xmax = parameters["axesrange"]["xmax"]
        ymin = parameters["axesrange"]["ymin"]
        ymax = parameters["axesrange"]["ymax"]
        ax2.set_xlim(xmin,xmax)
        ax2.set_ylim(ymin,ymax)

    # plot decision boundary
    # xx,yy,Z = decision_bound
    # Z = Z.reshape(xx.shape)
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

    # Set up formatting for the movie files
    # anim.save('data/hyperplane/z_plot.mp4', writer="ffmpeg")
    # anim2.save('data/hyperplane/x_plot.mp4',writer="ffmpeg")
    plt.show()


def animation(batchdata,parameters=None):
    # seaborn.set_style("whitegrid")
    # build plot
    fig, ax = plt.subplots()

    # build static plot
    x = batchdata[0]["xtrain"]
    y = batchdata[0]["ytrain"]
    cm_dark = mpl.colors.ListedColormap(getClassColors())
    sca = ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=cm_dark)
    if parameters:
        xmin = parameters["axesrange"]["xmin"]
        xmax = parameters["axesrange"]["xmax"]
        ymin = parameters["axesrange"]["ymin"]
        ymax = parameters["axesrange"]["ymax"]
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
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
