import os

import numpy as np

from kcipt.utils import data_gen_one


def matlab2np(arr):
    return np.array((np.array(arr._data).reshape(arr.size, order='F')), order='C')


def np2matlab(arr):
    import matlab.engine

    return matlab.double([[float(v) for v in row] for row in arr])


def kcit(x, y, z, seed=None, mateng=None):
    import matlab.engine

    if mateng is None:
        mateng = matlab.engine.start_matlab()
        dir_at = os.path.expanduser('~/Dropbox/research/2014 rcm/workspace/python/KCIPT2017/chsic')
        mateng.addpath(mateng.genpath(dir_at))

        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))

        v1, pvalue, statistic = mateng.hsiccondTestIC(x, y, z, 0.01, 1000, nargout=3)
        mateng.quit()

        return v1, pvalue, statistic
    else:
        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))

        v1, pvalue, statistic = mateng.hsiccondTestIC(x, y, z, 0.01, 1000, nargout=3)
        return v1, pvalue, statistic


if __name__ == '__main__':
    import matlab.engine

    mateng = matlab.engine.start_matlab()
    dir_at = os.path.expanduser('~/Dropbox/research/2014 rcm/workspace/python/KCIPT2017/chsic')
    mateng.addpath(mateng.genpath(dir_at))

    with open('chsic_one_50_i_seed.txt', 'a') as f:
        for slope in [0, 0.5, 1.0]:
            for i in range(1000):
                _, _, _, x, y, z = data_gen_one(50, i, slope=slope)
                v1, pvalue, statistic = kcit(np2matlab(x), np2matlab(y), np2matlab(z), mateng=mateng)
                print(slope, pvalue, file=f, sep=',')

    mateng.quit()
