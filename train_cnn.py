import tensorflow as tf
import utils
from dataset import Dataset
from model.build_cnn import cnn
from train_common import *

def report_test_accuracy(
    sess, images, labels, keep_prob, acc, dataset):
    batch_images, batch_labels = dataset.get_test()
    test_acc = sess.run(acc,
        feed_dict={images : batch_images, labels : batch_labels, keep_prob : 1})
    print('TEST ACCURACY:', test_acc)

def report_training_progress(
    sess, batch_index, images, labels, keep_prob, loss, acc, dataset):
    if batch_index % 50 == 0:
        batch_images, batch_labels = dataset.get_valid_batch(batch_size=512)
        test_acc, test_loss = sess.run(
            [acc, loss],
            feed_dict={images : batch_images, labels : batch_labels, keep_prob : 1})
        utils.log_training(batch_index, test_loss, test_acc)
        utils.update_training_plot(batch_index, test_acc, test_loss)

def train_cnn(
    sess, saver, save_path, images, labels, keep_prob, loss, train_op, acc, dataset):
    utils.make_training_plot()
    for batch_index in range(1700):
        report_training_progress(
            sess, batch_index, images, labels, keep_prob, loss, acc, dataset)
        # Run one step of training
        batch_images, batch_labels = dataset.get_train_batch(128)
        sess.run(train_op, feed_dict={images: batch_images, labels: batch_labels, keep_prob : 0.8})
        # Save model parameters periodically
        if batch_index % 50 == 0:
            saver.save(sess, save_path)

def main():
    print('building model...')
    images, labels, keep_prob = placeholders()
    logits = cnn(images, keep_prob)
    acc = accuracy(labels, logits)
    loss = cross_entropy_loss(labels, logits)
    train_op = optimizer(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver, save_path = utils.restore(sess, './checkpoints/cnn/')
        dataset = Dataset()
        report_test_accuracy(sess, images, labels, keep_prob, acc, dataset)
        train_cnn(
            sess, saver, save_path, images,
            labels, keep_prob, loss, train_op, acc, dataset)
        print('saving trained model...\n')
        saver.save(sess, save_path)
        utils.hold_training_plot()

if __name__ == '__main__':
    main()