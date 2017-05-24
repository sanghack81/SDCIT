import os
import time

import scipy.io

from sdcit.algo import c_KCIPT
from sdcit.utils import median_heuristic, K2D

if __name__ == '__main__':
    # experiments
    independent = 1
    for N in [200, 400]:
        for trial in range(3):
            mat_load = scipy.io.loadmat(os.path.expanduser('~/kcipt_data/{}_{}_{}_{}_chaotic.mat'.format('0.0', trial, independent, N)), squeeze_me=True, struct_as_record=False)
            data = mat_load['data']
            X = data.Yt1
            Y = data.Xt
            Z = data.Yt[:, 0: 2]

            start = time.time()

            kkk = median_heuristic(X, Y, Z)
            Dz = K2D(kkk[-1])
            # c_KCIPT(*kkk, Dz, B=25, b=10000, M=10000)
            c_KCIPT(*kkk, Dz, B=1000, b=50, M=0)

            endtime = time.time()
            print(endtime - start, trial, N, sep=',', flush=True)
