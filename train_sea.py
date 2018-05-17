# -*- coding: utf-8 -*-
"""
 Algorithm on sea data set
"""


from ANN import ANNModel
import preprocessing_for_sea as pd
import dataprocessing as dp
import time
from pandas import DataFrame
from visualize import visualize_out_prob

def main():
    # path = 'data/sea'
    # batchdata = pd.data_preprocessing(path,train_batchsize = 250,test_batchsize = 250)
    path = 'data/checkerboard_data'
    batchdata = pd.data_preprocessing(path,train_batchsize = 1024,test_batchsize = 25)
    parameters = dp.dataset_config('transientChessboard')

    start = time.clock()
    model = ANNModel()
    # z,decision_bound = model.model(batchdata,layer_structure=[3,2,2,2])
    z, decision_bound = model.model(batchdata, layer_structure=parameters['layer_structure'],
                                    LEARNING_RATE=parameters['learning_rate'], num_epochs=parameters['num_epochs'])
    visualize_out_prob(z,batchdata,decision_bound)
    end = time.clock()
    # print "time:%f"%(end-start)
    # pd.save_acc(acc)
    # print "average error rate:  %s %%" % (100*np.mean(acc))
    # plot_all_curve(homepath='', acc=acc)


if __name__ == '__main__':
    main()
