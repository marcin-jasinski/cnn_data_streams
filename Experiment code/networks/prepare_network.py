# -*- coding: utf-8 -*-
"""
Created on Mon Mar 8 2021

@author: Marcin
"""

from .normal_cnn import build_normal_cnn, build_normal_cnn_simple
from .bayesian_cnn import build_bayesian_cnn, build_bayesian_cnn_simple
from .monte_carlo_cnn import build_monte_carlo_cnn, build_monte_carlo_cnn_simple

def get_model(model_name, n_classes, original_dimensions, optimizer, chunk_size = 32, simple = False):

    if (not simple):
        if (model_name == 'normal_cnn'):
            return build_normal_cnn(n_classes, original_dimensions, optimizer)
        elif (model_name == 'monte_carlo_cnn'):
            return build_monte_carlo_cnn(n_classes, original_dimensions, optimizer)
        elif (model_name == 'bayesian_cnn'):
            return build_bayesian_cnn(n_classes, original_dimensions, optimizer, chunk_size)
        else:
            raise Exception('[ERROR] Unknown model name: ' + model_name)
    else:
        if (model_name == 'normal_cnn'):
            return build_normal_cnn_simple(n_classes, original_dimensions, optimizer)
        elif (model_name == 'monte_carlo_cnn'):
            return build_monte_carlo_cnn_simple(n_classes, original_dimensions, optimizer)
        elif (model_name == 'bayesian_cnn'):
            return build_bayesian_cnn_simple(n_classes, original_dimensions, optimizer, chunk_size)
        else:
            raise Exception('[ERROR] Unknown model name: ' + model_name)
