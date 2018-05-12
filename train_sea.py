# -*- coding: utf-8 -*-
"""
 Algorithm on sea data set
"""


from ANN import ANNModel
import preprocessing_for_sea as pd
import time
from pandas import DataFrame
from visualize import visualize_out_prob


def main():
    path = 'data/sea/'
    batchdata = pd.data_preprocessing(path,split=True)

    # print "total time step: %d" %(len(batchdata))

    start = time.clock()
    model = ANNModel()
    z,decision_bound = model.model(batchdata)
    visualize_out_prob(z,batchdata,decision_bound)
    end = time.clock()
    # print "time:%f"%(end-start)
    # pd.save_acc(acc)
    # print "average error rate:  %s %%" % (100*np.mean(acc))
    # plot_all_curve(homepath='', acc=acc)


if __name__ == '__main__':
    main()
