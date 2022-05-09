# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:04:01 2021

@author: Marcin
"""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

def get_optimizer(optimizer_name):

    SGD_opt = tf.keras.optimizers.SGD(lr=0.001)
    Adam_opt = tf.keras.optimizers.Adam(lr=0.001)
    Nadam_opt = tf.keras.optimizers.Nadam(lr=0.001)
    Adagrad_opt = tf.keras.optimizers.Adagrad(lr=0.001)
    Adamax_opt = tf.keras.optimizers.Adamax(lr=0.001)
    RMSprop_opt = tf.keras.optimizers.RMSprop(lr=0.001)

    return {
        'SGD': SGD_opt,
        'Adam': Adam_opt,
        'Nadam': Nadam_opt,
        'Adagrad': Adagrad_opt,
        'Adamax': Adamax_opt,
        'RMSprop': RMSprop_opt,
    }[optimizer_name]

def build_bayesian_cnn(n_classes, original_dimensions, optimizer, chunk_size = 32):

    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
      tf.cast(chunk_size, dtype=tf.float32))

    model = tf.keras.models.Sequential([
        tfp.layers.Convolution2DFlipout(32, 3, padding='SAME',
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.leaky_relu),
        tf.keras.layers.MaxPooling2D(2),
        tfp.layers.Convolution2DFlipout(64, 3, padding='SAME',
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.leaky_relu),
        tf.keras.layers.MaxPooling2D(2),
        tfp.layers.Convolution2DFlipout(128, 3, padding='SAME',
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.leaky_relu),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(64, kernel_divergence_fn=kl_divergence_function,
                                activation=tf.nn.leaky_relu),
        tfp.layers.DenseFlipout(n_classes, kernel_divergence_fn=kl_divergence_function,
                                activation=tf.nn.softmax)
    ])

    optimizer = get_optimizer(optimizer)

    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    return model

def build_bayesian_cnn_simple(n_classes, original_dimensions, optimizer, chunk_size = 32):

    kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
      tf.cast(chunk_size, dtype=tf.float32))

    model = tf.keras.models.Sequential([
        tfp.layers.Convolution2DFlipout(16, 5, padding='SAME',
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.leaky_relu),
        tf.keras.layers.MaxPooling2D(2),
        tfp.layers.Convolution2DFlipout(32, 5, padding='SAME',
                                        kernel_divergence_fn=kl_divergence_function,
                                        activation=tf.nn.leaky_relu),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseFlipout(16, kernel_divergence_fn=kl_divergence_function,
                                activation=tf.nn.leaky_relu),
        tfp.layers.DenseFlipout(n_classes, kernel_divergence_fn=kl_divergence_function,
                                activation=tf.nn.softmax)
    ])

    optimizer = get_optimizer(optimizer)

    model.compile(optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    return model
