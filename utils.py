import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def restore(sess, checkpoint_path):
    """
    If a checkpoint exists, restores the tensorflow model from the checkpoint.
    Returns the tensorflow Saver and the checkpoint filename.
    """
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    if checkpoint:
        path = checkpoint.model_checkpoint_path
        print('Restoring model parameters from {}'.format(path))
        saver.restore(sess, path)
    else:
        print('No saved model parameters found')
    # Return checkpoint path for call to saver.save()
    save_path = os.path.join(
        checkpoint_path, os.path.basename(os.path.dirname(checkpoint_path)))
    return saver, save_path

def log_training(batch_index, valid_loss, valid_acc=None):
    """
    Logs the validation accuracy and loss to the terminal
    """
    print('Batch {}'.format(batch_index))
    if valid_acc != None:
        print('\tCross entropy validation loss: {}'.format(valid_loss))
        print('\tAccuracy: {}'.format(valid_acc))
    else:
        print('\tMean squared error loss: {}'.format(valid_loss))

def make_training_plot():
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    plt.ion()
    plt.title('Supervised Network Training')
    plt.subplot(1, 2, 1)
    plt.xlabel('Batch Index')
    plt.ylabel('Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.xlabel('Batch Index')
    plt.ylabel('Validation Loss')

def make_ae_training_plot():
    """
    Runs the setup for an interactive matplotlib graph that logs the loss
    """
    plt.ion()
    plt.title('Autoencoder Training')
    plt.xlabel('Batch Index')
    plt.ylabel('Validation MSE')

def update_training_plot(batch_index, valid_acc, valid_loss):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    plt.subplot(1, 2, 1)
    plt.scatter(batch_index, valid_acc, c='b')
    plt.subplot(1, 2, 2)
    plt.scatter(batch_index, valid_loss, c='r')
    plt.pause(0.00001)

def update_ae_training_plot(batch_index, valid_loss):
    """
    Updates the training plot with a new data point for loss
    """
    plt.scatter(batch_index, valid_loss, c='r')
    plt.pause(0.00001)

def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()