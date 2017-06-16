import itertools
import os

import numpy as np
import scipy
import scipy.io

from experiments.exp_setup import SDCIT_DATA_DIR
from sdcit.utils import rbf_kernel_median, K2D, cythonize


def chaotic_configs():
    return list(itertools.product([0, 1], [200, 400], ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']))


def postnonlinear_noise_configs():
    return list(itertools.product(range(5), [0, 1], [200, 400])) + \
           list(itertools.product([9, 19, 49], [0, 1], [400]))  # high-dimensional


def read_chaotic(independent, gamma, trial, N, dir_at=SDCIT_DATA_DIR + '/'):
    X, Y, Z = read_chaotic_data(independent, gamma, trial, N, dir_at)
    kx, ky, kz = rbf_kernel_median(X, Y, Z)
    Dz = K2D(kz)
    return kx, ky, kz, Dz


def read_postnonlinear_noise(independent, noise, trial, N, dir_at=SDCIT_DATA_DIR + '/'):
    X, Y, Z = read_postnonlinear_noise_data(independent, noise, trial, N, dir_at)
    kx, ky, kz = rbf_kernel_median(X, Y, Z)

    dist_mat_file = os.path.expanduser(dir_at + 'dist_{}_{}_{}_{}_postnonlinear.mat'.format(noise, trial, independent, N))
    mat_load = scipy.io.loadmat(dist_mat_file, squeeze_me=True, struct_as_record=False)
    Dz = np.array(mat_load['D'])

    return cythonize(kx, ky, kz, Dz)


def read_chaotic_data(independent, gamma, trial, N, dir_at=SDCIT_DATA_DIR + '/'):
    mat_load = scipy.io.loadmat(os.path.expanduser(dir_at + '{}_{}_{}_{}_chaotic.mat'.format(gamma, trial, independent, N)), squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    if independent:
        X = data.Xt1
        Y = data.Yt
        Z = data.Xt[:, 0:2]
    else:
        X = data.Yt1
        Y = data.Xt
        Z = data.Yt[:, 0: 2]

    return X, Y, Z


def read_postnonlinear_noise_data(independent, noise, trial, N, dir_at=SDCIT_DATA_DIR + '/'):
    data_file = os.path.expanduser(dir_at + '{}_{}_{}_{}_postnonlinear.mat'.format(noise, trial, independent, N))
    mat_load = scipy.io.loadmat(data_file, squeeze_me=True, struct_as_record=False)
    data = mat_load['data']
    X = np.array(data.X).reshape((len(data.X), -1))
    Y = np.array(data.Y).reshape((len(data.Y), -1))
    Z = np.array(data.Z).reshape((len(data.Z), -1))
    return X, Y, Z
