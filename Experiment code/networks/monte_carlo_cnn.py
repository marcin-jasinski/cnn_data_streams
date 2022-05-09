# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:04:01 2021

@author: Marcin
"""
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras import layers

class MCDropout(layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

def build_monte_carlo_cnn(n_classes, original_dimensions, optimizer):

    model = Sequential([
        layers.Conv2D(32, 3, padding='same', activation="relu", input_shape = original_dimensions),
        layers.MaxPooling2D(2),
        MCDropout(0.4),
        layers.Conv2D(64, 3, padding='same', activation="relu"),
        layers.MaxPooling2D(2),
        MCDropout(0.4),
        layers.Conv2D(128, 3, padding='same', activation="relu"),
        layers.MaxPooling2D(2),
        MCDropout(0.4),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        MCDropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer=SGD(learning_rate=0.001) if optimizer == 'SGD' else Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.build()
    return model

def build_monte_carlo_cnn_simple(n_classes, original_dimensions, optimizer):

    model = Sequential([
        layers.Conv2D(16, 5, padding='same', activation="relu", input_shape = original_dimensions),
        layers.MaxPooling2D(2),
        MCDropout(0.4),
        layers.Conv2D(32, 5, padding='same', activation="relu"),
        layers.MaxPooling2D(2),
        MCDropout(0.4),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        MCDropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(optimizer=SGD(learning_rate=0.001) if optimizer == 'SGD' else Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.build()
    return model
