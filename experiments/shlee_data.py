import numpy as np
import scipy.io as sio
from tqdm import trange

from kcipt.utils import data_gen_one, data_gen_old


def np_to_nested_list(arr):
    return [[float(v) for v in row] for row in arr]


if __name__ == '__main__':
    for i in trange(300):
        for n in [50, 100]:
            for slope in [0, 0.5, 1, 2]:
                x, y, z = data_gen_one(n, i, slope, skip_kernel=True)
                sio.savemat('../data/one_i_{}_n_{}_slope_{}.mat'.format(i, n, slope), {'X': x, 'Y': y, 'Z': z})
                np.savez_compressed('../data/one_i_{}_n_{}_slope_{}.npz'.format(i, n, slope), X=x, Y=y, Z=z)

    for i in trange(300):
        for n in [50, 100]:
            for slope in [0, 0.5, 1, 2]:
                x, y, z = data_gen_old(n, i, slope, skip_kernel=True)
                sio.savemat('../data/old_i_{}_n_{}_slope_{}.mat'.format(i, n, slope), {'X': x, 'Y': y, 'Z': z})
                np.savez_compressed('../data/old_i_{}_n_{}_slope_{}.npz'.format(i, n, slope), X=x, Y=y, Z=z)
