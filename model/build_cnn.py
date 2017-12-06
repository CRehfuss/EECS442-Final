import numpy as np
import tensorflow as tf
from math import sqrt

def activation(layer, act_func):
    if not act_func:
        return layer
    elif act_func == 'relu':
        return tf.nn.relu(layer)
    elif act_func == 'tanh':
        return tf.nn.tanh(layer)

def conv_layer(layer_in, filter_size, channels_in, channels_out, stride, act_func):
    weights = tf.Variable(tf.truncated_normal(
        [filter_size, filter_size, channels_in, channels_out],
        mean=0,
        stddev=1/sqrt(filter_size*filter_size*channels_in)))
    bias = tf.Variable(tf.zeros(channels_out))
    layer = tf.nn.conv2d(layer_in, weights, [1, stride, stride, 1], padding='SAME')
    layer += bias
    return activation(layer, act_func)

def dense_layer(layer_in, channels_in, channels_out, act_func, keep_prob=None):
    weights = tf.Variable(tf.truncated_normal(
        [channels_in, channels_out],
        mean=0,
        stddev=1/sqrt(channels_in)))
    bias = tf.Variable(tf.zeros(channels_out))
    layer = tf.matmul(layer_in, weights) + bias
    if keep_prob is not None:
        return tf.nn.dropout(activation(layer, act_func), keep_prob)
    return activation(layer, act_func)

def cnn(X, keep_prob):
    layer = tf.image.rgb_to_grayscale(X)
    layer = conv_layer(layer, 5, 1, 64, 2, 'relu')
    layer = conv_layer(layer, 5, 64, 128, 2, 'relu')
    flat_layer = tf.reshape(layer, [-1, 8192])
    flat_layer = dense_layer(flat_layer, 8192, 4096, 'relu', keep_prob)
    flat_layer = dense_layer(flat_layer, 4096, 2048, 'relu', keep_prob)
    flat_layer = dense_layer(flat_layer, 2048, 1024, 'relu', keep_prob)
    flat_layer = dense_layer(flat_layer, 1024, 512, 'relu', keep_prob)
    flat_layer = dense_layer(flat_layer, 512, 256, 'relu', keep_prob)
    flat_layer = dense_layer(flat_layer, 256, 128, 'relu', keep_prob)
    flat_layer = dense_layer(flat_layer, 128, 62, None, keep_prob)
    return flat_layer
