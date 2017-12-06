import numpy as np
import tensorflow as tf
import utils
from model.build_cnn import cnn
from train_common import *

def predict_char(image):
    images, _, keep_prob = placeholders()
    logits = cnn(images, keep_prob)
    pred = predictions(logits)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver, save_path = utils.restore(sess, './checkpoints/cnn/')
        if not tf.train.get_checkpoint_state('./checkpoints/cnn/'):
            raise Error('No checkpoint found')
        guess = sess.run(pred, feed_dict={images: image, keep_prob: 1})
        return guess