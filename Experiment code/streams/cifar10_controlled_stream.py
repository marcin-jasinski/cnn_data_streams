# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:20:31 2021

@author: Marcin

Klasy 1 i 9: automobile i truck
Klasy 4 i 7: deer i horse
"""
import numpy as np

from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.utils import shuffle
from skmultiflow.data import DataStream

def prepare_controlled_cifar10_stream(samples_per_class,
                                      forgetting_mode,
                                      class_set_A = (1, 4),
                                      class_set_B = (7, 9)):
    # Wczytanie zbioru danych
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    # Normalizacja
    original_dimensions = X.shape[1], X.shape[2], X.shape[3]
    X_reshape = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    X_norm = X_reshape.astype(np.float32) / 255.0

    # Spłaszczenie na potrzeby generatora danych strumieniowych
    X_flat = X_norm.reshape((X.shape[0], -1))

    # Liczba klas w datasetcie
    _, N_DISTINCT_CLASSES = np.unique(y, return_counts=True)

    if (samples_per_class > np.min(N_DISTINCT_CLASSES)
        or samples_per_class > np.max(N_DISTINCT_CLASSES)):
        raise Exception('[ERROR] Number of classes exceeds number of available samples in dataset!')

    stream_X = np.array([])
    stream_y = np.array([])

    class_pairs = [class_set_A, class_set_B]
    if (forgetting_mode):
        class_pairs.append(class_set_A)

    for class_set in range(0, len(class_pairs)):
        y_indx_A = [i for i, cl in enumerate(y) if cl == class_pairs[class_set][0]]
        y_indx_A = y_indx_A[:samples_per_class]

        X_samples_A = np.array(X_flat[y_indx_A])
        y_samples_A = np.array(y[y_indx_A])

        y_indx_B = [i for i, cl in enumerate(y) if cl == class_pairs[class_set][1]]
        y_indx_B = y_indx_B[:samples_per_class]

        X_samples_B = np.array(X_flat[y_indx_B])
        y_samples_B = np.array(y[y_indx_B])

        # Konkatenacja dwóch strumieni klas
        stream_task_X = np.concatenate((X_samples_A, X_samples_B), axis=0)
        stream_task_y = np.concatenate((y_samples_A, y_samples_B), axis=0)

        # Pomieszanie próbek
        stream_task_X, stream_task_y = shuffle(stream_task_X, stream_task_y)

        if not stream_X.size:
            stream_X = stream_task_X
            stream_y = stream_task_y
        else:
            stream_X = np.concatenate((np.asarray(stream_X), stream_task_X), axis=0)
            stream_y = np.concatenate((np.asarray(stream_y), stream_task_y), axis=0)

    stream_y = to_categorical(stream_y)

    cifar10_stream = DataStream(stream_X, stream_y)
    n_samples = stream_y.shape[0]
    n_classes = stream_y.shape[1]

    return cifar10_stream, n_samples, n_classes, original_dimensions
