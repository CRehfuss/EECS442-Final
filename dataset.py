import numpy as np
from scipy.misc import imread, imresize
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import os

class Dataset:

    def __init__(self):
        self.data = loadmat('lists.mat')['list']
        self.images = self._get_images()
        self.labels = np.asscalar(self.data['ALLlabels']-1).reshape(-1)
        self.train_images, self.test_images, self.train_labels, self.test_labels = \
            train_test_split(self.images, self.labels, test_size = 0.15)

    def get_train_batch(self, batch_size=32):
        idx = np.random.choice(len(self.train_images), size=batch_size, replace=False)
        return self.train_images[idx], self.train_labels[idx]

    def get_test_batch(self, batch_size=32):
        idx = np.random.choice(len(self.test_images), size=batch_size, replace=False)
        return self.test_images[idx, :, :, :], self.test_labels[idx]    

    def _get_images(self):
        images = []
        image_paths = np.asscalar(self.data['ALLnames'])
        for image_path in image_paths:
            image = imresize(imread(image_path + '.png'), (32, 32, 3))
            if image.shape != (32, 32, 3):
                image = np.stack((image,)*3, axis=2)
            images.append(image)
        return np.array(images)

    def _get_label_name(self, label):
        if label < 10:
            return str(label)
        if label < 36:
            return chr(ord('A') - 10 + label)
        return chr(ord('a') - 36 + label)
