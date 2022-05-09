# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:19:08 2021

@author: Marcin
"""
import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical
from skmultiflow.data import DataStream

def prepare_noisy_mnist_stream(stream_length, noise_start, noise_length, noise_factor=0.3):

    if (noise_start + noise_length > stream_length):
        raise Exception('[ERROR] Incorrect stream parameters!')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    X = X[:stream_length]
    y = y[:stream_length]

    # Normalizacja
    original_dimensions = X.shape[1], X.shape[2], 1

    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
    X = X.astype(np.float32) / 255.0

    # Szum
    for i in range(noise_start, noise_start+noise_length+1):
        X[i] = X[i] + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X[i].shape)
        X[i] = np.clip(X[i], 0., 1.)

    X_flat = X.reshape((X.shape[0], -1))
    y = to_categorical(y)
    n_samples = y.shape[0]
    n_classes = y.shape[1]

    mnist10_stream = DataStream(X_flat, y)

    return mnist10_stream, n_samples, n_classes, original_dimensions
