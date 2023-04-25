import os
import glob
import cv2 as cv
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import sklearn.neighbors as nn
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.signal import gaussian, convolve
from scipy.ndimage import gaussian_filter1d

def load_data(size=64):
    image_folder = '../../Datasets/tiny_imagenet/tiny-imagenet-200/test/images'
    filenames = glob.glob(os.path.join(image_folder, '*.jpeg'))
    filenames = np.random.choice(filenames, size=min(100000, len(filenames)), replace=False)
    
    X_ab = np.empty((len(filenames), size, size, 2))
    
    for i, filename in enumerate(filenames):
        bgr = cv.resize(cv.imread(filename), (size, size), cv.INTER_CUBIC)
        lab = cv.cvtColor(bgr, cv.COLOR_BGR2LAB).astype(np.int32)
        X_ab[i] = lab[:, :, 1:] - 128
    
    return X_ab


def compute_color_prior(X_ab, size=64, do_plot=False):
    q_ab = np.load(os.path.join(data_dir, "pts_in_hull.npy"))

    if do_plot:
        plt.figure(figsize=(15, 15))
        plt.scatter(q_ab[:, 0], q_ab[:, 1])
        for i in range(q_ab.shape[0]):
            plt.text(q_ab[i, 0], q_ab[i, 1], str(i), fontsize=6)
        plt.xlim([-110, 110])
        plt.ylim([-110, 110])
        plt.show()
        
        plt.hist2d(X_ab[..., 0].ravel(), X_ab[..., 1].ravel(), bins=100, density=True, norm=LogNorm())
        plt.xlim([-110, 110])
        plt.ylim([-110, 110])
        plt.colorbar()
        plt.show()

    X_ab = X_ab.reshape(-1, 2)
    nearest = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(q_ab)
    _, ind = nearest.kneighbors(X_ab)
    counts = np.bincount(ind.ravel())
    prior_prob = np.zeros(q_ab.shape[0])
    np.put(prior_prob, np.nonzero(counts)[0], counts[np.nonzero(counts)])
    prior_prob /= np.sum(prior_prob)
    np.save(os.path.join(data_dir, "prior_prob.npy"), prior_prob)

    if do_plot:
        plt.hist(prior_prob, bins=100)
        plt.yscale("log")
        plt.show()

def smooth_color_prior(size=64, sigma=5, do_plot=False):
    prior_prob = np.load(os.path.join(data_dir, "prior_prob.npy"))
    prior_prob += 1E-3 * np.min(prior_prob)
    prior_prob_smoothed = gaussian_filter1d(prior_prob, sigma, mode='nearest')
    prior_prob_smoothed /= prior_prob_smoothed.sum()
    np.save(os.path.join(data_dir, "prior_prob_smoothed.npy"), prior_prob_smoothed)
    if do_plot:
        plt.plot(prior_prob)
        plt.plot(prior_prob_smoothed, "g--")
        plt.yscale("log")
        plt.show()



def compute_prior_factor(size=64, gamma=0.5, alpha=1, do_plot=False):
    prior_prob_smoothed = np.load(os.path.join(data_dir, "prior_prob_smoothed.npy"))
    u = np.ones_like(prior_prob_smoothed) / prior_prob_smoothed.size
    prior_factor = ((1 - gamma) * prior_prob_smoothed + gamma * u) ** (-alpha)
    prior_factor /= np.sum(prior_factor * prior_prob_smoothed)
    np.save(os.path.join(data_dir, "prior_factor.npy"), prior_factor)
    if do_plot:
        plt.plot(prior_factor)
        plt.yscale("log")
        plt.show()


if __name__ == '__main__':
    data_dir = 'data/'
    do_plot = True

    X_ab = load_data()
    compute_prior_factor(do_plot=True)
