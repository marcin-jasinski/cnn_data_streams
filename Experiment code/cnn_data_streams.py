# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 2021

@author: Marcin
"""

from streams.prepare_streams import make_data_stream
from utils.utils import make_results_dir, train_models_on_streams

import os

# set up tensorflow backend for keras
os.environ['KERAS_BACKEND'] = 'tensorflow'


if __name__ == "__main__":

    EPOCHS = 50
    CHUNK_SIZE = 32

    results_save = True
    results_dir_name = "__results__"
    results_dir = make_results_dir(results_dir_name, results_save)

    models = {
        'Normal': 'normal_cnn',
        'MonteCarlo': 'monte_carlo_cnn',
        'Bayesian': 'bayesian_cnn'
    }

    optimizers = ['SGD', 'Adam']
    simple_model = False

    datasets = ['cifar10', 'mnist']
    stream_leangths = [6000, 12000]
    cifar_forgetting_modes = [False, True]

    for model_name in models:
        for optimizer_name in optimizers:
            for dataset_name in datasets:
                for stream_len in stream_leangths:
                    if (dataset_name == 'cifar10'):
                        for cifar_forgetting in cifar_forgetting_modes:
                            stream, n_samples, n_classes, original_dimensions = make_data_stream(dataset_name,
                                                                                                 stream_len,
                                                                                                 cifar_forgetting)
                            train_models_on_streams(stream, n_samples, n_classes,
                                                    models, model_name, simple_model, optimizer_name,
                                                    dataset_name, original_dimensions, cifar_forgetting,
                                                    EPOCHS, CHUNK_SIZE, results_dir)
                    else:
                        stream, n_samples, n_classes, original_dimensions = make_data_stream(dataset_name,
                                                                                             stream_len, False)
                        train_models_on_streams(stream, n_samples, n_classes,
                                                models, model_name, simple_model, optimizer_name,
                                                dataset_name, original_dimensions, False,
                                                EPOCHS, CHUNK_SIZE, results_dir)
