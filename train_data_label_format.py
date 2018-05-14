# -*- coding: utf-8 -*-
"""
 Algorithm on sea data set
"""

import dataprocessing as dp
from visualize import animation
from visualize import visualize_out_prob
from ANN import ANNModel


def visualze_only(batchdata,parameters):
    animation(batchdata,parameters)


def train_and_visualize(batchdata,parameters):
    model = ANNModel()
    z, decision_bound = model.model(batchdata, learning_rate=parameters['learning_rate'],
                                    num_epochs=parameters['num_epochs'])
    # visualize_out_prob(z, batchdata, decision_bound, parameters)

def main():
    parameters = dp.dataset_config('movingSquares')
    #parameters = dp.dataset_config('rotatingHyperplane')
    # data preprocessing including data reading, batch splitting and probable data augmenting
    batchdata = dp.data_preprocessing(parameters["outpath"], batchsize = parameters["batchsize"])
    train_and_visualize(batchdata,parameters)
    # visualze_only(batchdata,parameters)

if __name__ == '__main__':
    main()
