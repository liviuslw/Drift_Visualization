# -*- coding: utf-8 -*-
"""
 Algorithm on sea Data set
"""
import numpy as np
import preprocessing_for_sea as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.svm import SVC
import seaborn

def animation(accuracy):
    seaborn.set_style("whitegrid")
    # build plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax2 = fig.add_subplot(1,1,1)
    # build static plot
    ax.set_xlim(1,200)
    ax.set_ylim(min(accuracy)-0.1,1)
    curve, = plt.plot([],[],lw=1.5,c='navy')
    plt.plot([],[],lw=2,c='red')
    ax.set_xlabel('time step')
    ax.set_ylabel('accuracy values')
    plt.title('time step {0}'.format(0))
    # build dynamic plot
    def update(i):
        x = range(1,i+1)
        y = accuracy[:i]
        curve.set_data(x,y)
        plt.title('time step {0}'.format(i+1))
        if i==55 or i==105 or i==155:
            x_end = (i-5)* np.ones(1000)
            y_end = np.linspace(0,accuracy[i-5],1000)
            lines = plt.plot(x_end,y_end,'r--')
            plt.pause(0.5)
        if i == len(accuracy):
            plt.pause(10)
    anim = FuncAnimation(fig, update,frames= range(1,len(accuracy)+1),interval=50)
    anim.save('visualize by acc.mp4', writer="ffmpeg")
    plt.show()


def train_then_test(batchdata):
    clf = SVC()
    clf.fit(batchdata[0]["xtrain"],batchdata[0]["ytrain"])
    acc = []
    for data in batchdata[1:]:
        acc.append(clf.score(data["xtest"],data["ytest"]))
    return acc

def main():
    path = 'Data/sea'
    batchdata = pd.data_preprocessing(path,train_batchsize=250,test_batchsize=250)
    acc = train_then_test(batchdata)
    animation(acc)


if __name__ == '__main__':
    main()