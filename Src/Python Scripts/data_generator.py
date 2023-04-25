import os
import random
from random import shuffle

import cv2 as cv
import numpy as np
import sklearn.neighbors as nn
from tensorflow.keras.utils import Sequence
import glob
from config import batch_size, img_rows, img_cols, nb_neighbors

image_folder = '../../Datasets/mini_imagenet/imagenet-mini/train_images'


def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w, _ = image_ab.shape
    ab = image_ab[:, :, :2].reshape(-1, 2)
    dist_neigh, idx_neigh = nn_finder.kneighbors(ab)
    wts = np.exp(-dist_neigh ** 2 / (2 * 5 ** 2))
    y = np.zeros((ab.shape[0], nb_q))
    y[np.arange(ab.shape[0])[:, None], idx_neigh] = wts / wts.sum(axis=1)[:, None]
    return y.reshape(h, w, nb_q)


class DataGenSequence(Sequence):
    def __init__(self, usage):
        self.usage = usage
        with open(f"../Helper_Data/{usage}_names.txt", "r") as f:
            self.names = f.read().splitlines()
        np.random.shuffle(self.names)
        q_ab, self.nb_q = np.load("../Helper_Data/colour_spaces.npy"), q_ab.shape[0]
        self.nn_finder = nn.NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __len__(self):
        N = len(self.names)/batch_size
        N = np.ceil(N)
        return int(N)

    def __getitem__(self, idx):
        i, length = idx * batch_size, min(batch_size, len(self.names) - idx * batch_size)
        batch_x, batch_y = np.empty((length, img_rows, img_cols, 1), dtype=np.float32), np.empty((length, img_rows // 4, img_cols // 4, self.nb_q), dtype=np.float32)
        for i_batch, name in enumerate(self.names[i:i+length]):
            bgr, gray = cv.imread(os.path.join(image_folder, name)), cv.imread(os.path.join(image_folder, name), 0)
            x = cv.resize(gray / 255., (img_rows, img_cols), cv.INTER_CUBIC)
            out_lab = cv.resize(cv.cvtColor(bgr, cv.COLOR_BGR2LAB), (img_rows // 4, img_cols // 4), cv.INTER_CUBIC)
            out_ab = out_lab[:, :, 1:].astype(np.int32) - 128
            y = get_soft_encoding(out_ab, self.nn_finder, self.nb_q)
            if np.random.random_sample() > 0.5:
                x, y = np.fliplr(x), np.fliplr(y)
            batch_x[i_batch, :, :, 0], batch_y[i_batch] = x, y

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)


def train_gen():
    return DataGenSequence('train')


def val_gen():
    return DataGenSequence('val')


def split_data():
    names = glob.glob(os.path.join(image_folder, '*.jpeg'))
    names = [os.path.basename(name) for name in names]


    num_samples, num_train_samples = len(names), int(len(names) * 0.9)
    num_val_samples, val_names = num_samples - num_train_samples, random.sample(names, num_samples - num_train_samples)
    train_names = list(set(names) - set(val_names))
    shuffle(val_names),shuffle(train_names)

    print('num_samples: ' + str(num_samples) + '\nnum_train_samples: ' + str(num_train_samples) + '\nnum_val_samples: ' + str(num_val_samples))
    
    with open('../Helper_Data/val_names.txt', 'w') as f1, open('../Helper_Data/train_names.txt', 'w') as f2:
        f1.write('\n'.join(val_names)), f2.write('\n'.join(train_names))


if __name__ == '__main__':
    split_data()
