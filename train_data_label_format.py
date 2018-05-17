# -*- coding: utf-8 -*-
"""
 Algorithm on sea data set
"""

import dataprocessing as dp
from visualize import animation
from visualize import visualize_out_prob
from ANN import ANNModel
from visualize import heap_map_plot

def visualze_only(batchdata,parameters):
    animation(batchdata,parameters)


def train_and_visualize(batchdata,parameters):
    model = ANNModel()
    z, decision_bound = model.model(batchdata, layer_structure = parameters['layer_structure'],
                                    LEARNING_RATE=parameters['learning_rate'], num_epochs=parameters['num_epochs'])
    visualize_out_prob(z, batchdata, decision_bound, parameters)


def main():
    # parameters = dp.dataset_config('rotatingHyperplane')
    # parameters = dp.dataset_config('movingRBF')
    # parameters = dp.dataset_config('interchangingRBF')
    # parameters = dp.dataset_config('movingSquares')
    # parameters = dp.dataset_config('weather')
    # parameters = dp.dataset_config('Elec2')
    parameters = dp.dataset_config('covType')
    # parameters = dp.dataset_config('poker')
    # data preprocessing including data reading, batch splitting and probable data augmenting
    batchdata = dp.data_preprocessing(parameters["outpath"], batchsize = parameters["batchsize"])

    # heap map helps to find two features that strongly correlate with labels
    # heap_map_plot(batchdata[0])
    train_and_visualize(batchdata,parameters)
    # visualze_only(batchdata,parameters)

if __name__ == '__main__':
    main()
