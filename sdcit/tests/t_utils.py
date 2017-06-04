import numpy as np
import scipy.stats


def aupc(pvals):
    ttt = [(uniq_v, np.mean(pvals <= uniq_v)) for uniq_v in np.unique(pvals)]
    area = 0
    prev_x, prev_y = 0, 0
    for x, y in ttt:
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y
    area += (1 - prev_x) * prev_y
    return area


def ks_statistic(pvals):
    D, _ = scipy.stats.kstest(pvals, 'uniform')
    return D
