# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 2021

@author: Marcin
"""

from .cifar10_controlled_stream import prepare_controlled_cifar10_stream
from .cifar10_random_stream import prepare_random_cifar10_stream
from .mnist_noisy_stream import prepare_noisy_mnist_stream

def make_data_stream(dataset_name, stream_length, cifar_forgetting):

    cifar_samples_per_class = stream_length // 4
    mnist_noise_start_point = stream_length // 2
    mnist_noise_duration = stream_length // 4

    if (dataset_name == 'cifar10_random'):
        stream, n_samples, n_classes, original_dimensions = prepare_random_cifar10_stream(
            samples_per_class = cifar_samples_per_class, forgetting_mode = cifar_forgetting)
    elif (dataset_name == 'cifar10'):
        stream, n_samples, n_classes, original_dimensions = prepare_controlled_cifar10_stream(
            samples_per_class = cifar_samples_per_class, forgetting_mode = cifar_forgetting)
    elif (dataset_name == 'mnist'):
        stream, n_samples, n_classes, original_dimensions = prepare_noisy_mnist_stream(
            stream_length = stream_length,
            noise_start = mnist_noise_start_point,
            noise_length = mnist_noise_duration,
            noise_factor = 0.5)
    else:
        raise Exception('[ERROR] Incorrect dataset name')

    return stream, n_samples, n_classes, original_dimensions
