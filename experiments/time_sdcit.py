import os
import time

import scipy.io

from kcipt.sdcit import SDCIT
from kcipt.utils import median_heuristic, K2D

if __name__ == '__main__':
    with open('../results/sdcit_time.csv', 'w') as f:
        for N in [200, 400]:
            for b in [250, 500]:
                for trial in range(300):
                    mat_load = scipy.io.loadmat(os.path.expanduser('~/kcipt_data/{}_{}_{}_{}_chaotic.mat'.format('0.0', trial, 0, N)), squeeze_me=True, struct_as_record=False)
                    data = mat_load['data']
                    X = data.Yt1
                    Y = data.Xt
                    Z = data.Yt[:, 0: 2]

                    start = time.time()
                    kkk400 = median_heuristic(X, Y, Z)
                    D400 = K2D(kkk400[-1])
                    SDCIT(*kkk400, Dz=D400, size_of_null_sample=b, reserve_perm=True, with_post=True)
                    endtime = time.time()
                    print(endtime - start, trial, N, b, file=f, sep=',', flush=True)
