# -*- coding: utf-8 -*-
"""
 Algorithm on sea data set
"""


import ANN
import preprocessing_for_sea as pd
import time
from pandas import DataFrame


def main():
    path = 'data/sea/'
    batchdata = pd.data_preprocessing(path,split=False)

    # print "total time step: %d" %(len(batchdata))

    start = time.clock()
    # parameters,costs = ANN.model(batchdata)
    parameters, costs = ANN.instance_learning(batchdata)
    # data = DataFrame(costs)
    # data.to_csv('Data.csv', header='False', index='False')
    end = time.clock()
    # print "time:%f"%(end-start)
    # pd.save_acc(acc)
    # print "average error rate:  %s %%" % (100*np.mean(acc))
    # plot_all_curve(homepath='', acc=acc)


if __name__ == '__main__':
    main()
