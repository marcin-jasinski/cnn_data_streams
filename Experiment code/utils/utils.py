# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:26:59 2021

@author: Marcin
"""
import matplotlib.pyplot as plt
import pandas as pd
import os

from networks.prepare_network import get_model
from tqdm import tqdm
import numpy as np
import math

from datetime import datetime
from .network_utils import predict_class, predict_proba, calculate_uncertainty_metrics
from .network_utils import calc_max_perf_loss, calc_restoration_time

def train_models_on_streams(stream, stream_len, n_classes,
                            models, model_name, simple_model, optimizer_name,
                            dataset_name, original_dimensions, cifar_forgetting,
                            train_epochs, chunk_size, results_dir):

    # stream.prepare_for_use()
    training_steps = math.floor(stream_len / chunk_size)
    true_stream_length = training_steps * chunk_size

    # Get model
    model = get_model(models[model_name], n_classes,
                      original_dimensions, optimizer_name, chunk_size, simple=simple_model)

    # This step is required because of Keras API limitations
    train_sample, train_label = stream.next_sample()
    train_sample = train_sample.reshape(1, *original_dimensions)
    model.fit(train_sample, train_label, epochs=1,
              batch_size=chunk_size, verbose=0)
    # model.summary()

    print('\n<===========================>')
    print('[INFO] Model:     {}'.format(model_name))
    print('[INFO] Optimizer: {}'.format(optimizer_name))
    print('[INFO] Dataset:   {}'.format(dataset_name))
    print('[INFO] Samples in data stream: {}'.format(stream_len))
    print('[INFO] Training steps: {}'.format(training_steps))
    print('[INFO] Chunk size: {}'.format(chunk_size))

    acc_dist = []
    entropy_dist = []
    softmax_average_dist = []
    softmax_variance_dist = []

    # Training loop
    for i in tqdm(range(training_steps)):
        train_samples, train_labels = stream.next_sample(chunk_size)
        train_samples = train_samples.reshape(
            train_samples.shape[0], *original_dimensions)

        predictions = predict_proba(train_samples, model, 100)
        y_pred = predict_class(predictions)

        entropy, softmax_average, softmax_variance = calculate_uncertainty_metrics(predictions)

        entropy_dist.append(np.mean(entropy))
        softmax_average_dist.append(np.mean(softmax_average))
        softmax_variance_dist.append(np.mean(softmax_variance))

        labels_flat = np.argmax(train_labels, axis=1)
        acc = np.mean(y_pred == labels_flat)
        acc_dist.append(acc)

        model.fit(train_samples, train_labels, epochs=train_epochs, batch_size=chunk_size, verbose=0)

    plot_results(model_name, optimizer_name, dataset_name, cifar_forgetting, stream_len,
                 true_stream_length, training_steps, acc_dist,
                 entropy_dist, softmax_average_dist, softmax_variance_dist)

    save_results(results_dir, true_stream_length, stream_len, training_steps, model_name, optimizer_name,
                 dataset_name, cifar_forgetting, acc_dist,
                 entropy_dist, softmax_average_dist, softmax_variance_dist)


def calc_mean_results(acc_dist, training_steps, n_samples):
    mean_acc = [sum(acc_dist[:i]) / len(acc_dist[:i]) for i in range(1, training_steps+1)]
    return mean_acc


def calc_uncertainty_results(entropy_dist, training_steps):
    mean_entropy = [sum(entropy_dist[:i]) / len(entropy_dist[:i]) for i in range(1, training_steps+1)]
    return mean_entropy


def plot_results(model_name, optimizer_name,
                 dataset_name, cifar_forgetting,
                 stream_len, n_samples, training_steps,
                 acc_dist,
                 entropy_dist,
                 softmax_average_dist,
                 softmax_variance_dist):

    mean_acc = calc_mean_results(acc_dist, training_steps, n_samples)
    mean_entropy = calc_uncertainty_results(entropy_dist, training_steps)

    fig, axs = plt.subplots(3, 2, figsize=(6, 7))
    fig.suptitle('Accuracy, entropy and softmax values for model "{}" ({}) '.format(model_name, optimizer_name))

    time_acc_entropy = [i for i in range(0, training_steps)]

    axs[0, 0].plot(time_acc_entropy, acc_dist)
    axs[0, 0].set_title('Accuracy')
    axs[0, 1].plot(time_acc_entropy, mean_acc)
    axs[0, 1].set_title('Mean Accuracy')

    axs[1, 0].plot(time_acc_entropy, entropy_dist)
    axs[1, 0].set_title('Entropy')
    axs[1, 1].plot(time_acc_entropy, mean_entropy)
    axs[1, 1].set_title('Mean Entropy')

    axs[2, 0].plot(time_acc_entropy, softmax_average_dist)
    axs[2, 0].set_title('Softmax Average')
    axs[2, 1].plot(time_acc_entropy, softmax_variance_dist)
    axs[2, 1].set_title('Softmax Variance')

    plt.tight_layout()

    if (dataset_name != 'mnist'):
        dataset_name = dataset_name + '_' + str(cifar_forgetting)

    save_params = [model_name, optimizer_name, dataset_name, stream_len]
    plt.savefig('wykresy/{}_{}_{}_{}.png'.format(*save_params))
    plt.close()


def make_results_dir(results_dir_path, results_save):
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    results_dir = os.path.join(results_dir_path, timestamp)
    if(results_save):
        os.makedirs(results_dir)

    return results_dir


def save_results(results_dir,
                 n_samples, stream_len,
                 training_steps, model_name, optimizer_name,
                 dataset_name, cifar_forgetting,
                 acc_dist,
                 entropy_dist,
                 softmax_average_dist,
                 softmax_variance_dist):

    mean_acc = calc_mean_results(acc_dist, training_steps, n_samples)
    mean_entropy = calc_uncertainty_results(entropy_dist, training_steps)

    percentage_loss, point_percents = calc_max_perf_loss(acc_dist, training_steps)
    restoration_time = calc_restoration_time(acc_dist, training_steps)

    if not restoration_time:
        restoration_time.append(0)

    acc_df = pd.DataFrame(acc_dist)
    mean_acc_df = pd.DataFrame(mean_acc)

    entropy_df = pd.DataFrame(entropy_dist)
    mean_entropy_df = pd.DataFrame(mean_entropy)

    softmax_average_df = pd.DataFrame(softmax_average_dist)
    softmax_variance_df = pd.DataFrame(softmax_variance_dist)

    metrics = {'percentage_loss': percentage_loss,
               'point_percentage': point_percents,
               'restoration_time': restoration_time}

    metrics_df = pd.DataFrame(metrics)

    if (dataset_name != 'mnist'):
        dataset_name = dataset_name + '_' + str(cifar_forgetting)

    save_params = [model_name, optimizer_name, dataset_name, stream_len]

    acc_df.to_csv(os.path.join(
        results_dir, '{}_{}_{}_{}_accuracy.csv'.format(*save_params)))
    mean_acc_df.to_csv(os.path.join(
        results_dir, '{}_{}_{}_{}_mean_acc.csv'.format(*save_params)))

    entropy_df.to_csv(os.path.join(
        results_dir, '{}_{}_{}_{}_entropy.csv'.format(*save_params)))
    mean_entropy_df.to_csv(os.path.join(
        results_dir, '{}_{}_{}_{}_mean_entropy.csv'.format(*save_params)))

    softmax_average_df.to_csv(os.path.join(
        results_dir, '{}_{}_{}_{}_softmax_average.csv'.format(*save_params)))
    softmax_variance_df.to_csv(os.path.join(
        results_dir, '{}_{}_{}_{}_softmax_variance.csv'.format(*save_params)))

    metrics_df.to_csv(os.path.join(
        results_dir, '{}_{}_{}_{}_metrics.csv'.format(*save_params)))
