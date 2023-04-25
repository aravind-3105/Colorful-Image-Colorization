import multiprocessing

import cv2 as cv
import keras.backend as K
import numpy as np
from tensorflow.python.client import device_lib

# Load the color prior factor that encourages rare colors
prior_factor = np.load("../Helper_Data/prior_factor.npy")
prior_factor = prior_factor.astype(np.float32)

def categorical_crossentropy_color(y_true, y_pred):
    q = 313
    y_true,y_pred = K.reshape(y_true, (-1, q)), K.reshape(y_pred, (-1, q))
    idx_max = K.argmax(y_true, axis=1)
    weights = K.gather(prior_factor, idx_max)

    return K.mean(K.categorical_crossentropy(y_pred, y_true * K.reshape(weights, (-1, 1))), axis=-1)

# getting the number of GPUs
def get_available_gpus():
    return [x.name for x in device_lib.list_local_devices() if 'GPU' in x.device_type]


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()