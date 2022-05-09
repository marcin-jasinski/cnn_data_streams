# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 20:26:22 2021

@author: Marcin
"""
import cv2
import numpy as np

from PIL import Image
from keras import backend as K
from keras.preprocessing import image

def calc_restoration_time(acc_dist, num_steps):
    threshold = 0.95
    peak_index, peak_value = max(enumerate(acc_dist), key=lambda p: p[1])
    acc_drop_index, acc_drop_value = min(enumerate(acc_dist[peak_index:]), key=lambda p: p[1])
    restoration_value = next((x for x in acc_dist[acc_drop_index + peak_index:]
                              if x > peak_value * threshold), num_steps)

    restoration_index = np.where(np.asarray(acc_dist[peak_index:]) == restoration_value)[0]
    restoration_time = (restoration_index - peak_index) / num_steps
    return restoration_time

def calc_max_perf_loss(acc_dist, num_steps):
    peak_index, peak_value = max(enumerate(acc_dist), key=lambda p: p[1])
    acc_drop_index, acc_drop_value = min(enumerate(acc_dist[peak_index:]), key=lambda p: p[1])
    percentage_loss = 100 - ((acc_drop_value / peak_value) * 100)
    point_percents = (peak_value - acc_drop_value) * 100
    return percentage_loss, point_percents

def predict_proba(X, model, num_samples):
    preds = [model.predict(X) for _ in range(num_samples)]
    # return np.stack(preds).mean(axis=0)
    return preds

def predict_class(predictions):
    return np.argmax(np.mean(predictions, axis=0), axis=-1)

"""
Following entropy calculations are based on Rob Romijnder's solution
Source: https://github.com/RobRomijnders/weight_uncertainty
"""
def reduce_entropy(X, axis=-1):
    return -1 * np.sum(X * np.log(X+1E-12), axis=axis)

def calculate_uncertainty_metrics(preds, labels=None):
    if isinstance(preds, list):
        preds = np.stack(preds)

    # preds in shape [num_runs, num_batch, num_classes]
    num_runs, num_batch = preds.shape[:2]

    ave_preds = np.mean(preds, axis=0)
    pred_class = np.argmax(ave_preds, axis=1)

    # entropy of the posterior predictive
    entropy = reduce_entropy(ave_preds, axis=1)

    # Variance of softmax for the predicted class
    softmax_average = np.mean(preds[:, range(num_batch), pred_class], 0)
    softmax_variance = np.std(preds[:, range(num_batch), pred_class], 0)

    return entropy, softmax_average, softmax_variance

def make_heatmap(sample, model, i, target_class):
    data = sample
    data = np.reshape(data, (32,32,3), order='F' ) # Fortran-like indexing order
    data = np.expand_dims(data, axis=0)
    # data = preprocess_input(data)
    predict_single = model.predict(data)
    pred_max = np.argmax(predict_single[0])

    out = model.output[:, pred_max]
    last_conv_layer = model.get_layer(index=14)
    grads = K.gradients(out, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0,1,2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([data])

    for k in range (256):
        conv_layer_output_value[:, :, k] *= pooled_grads_value[k]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img_var = Image.fromarray(sample.astype(np.uint8))
    img = image.img_to_array(img_var)
    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.2 + img
    cv2.imwrite('heatmap_{}_{}.jpg'.format(i, target_class), superimposed_img)
