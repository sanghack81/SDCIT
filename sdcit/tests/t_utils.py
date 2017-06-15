from typing import List, Union

import numpy as np
import scipy.stats


def AUPC(pvals: Union[List, np.ndarray]) -> float:
    """Area Under Power Curve"""
    pvals = np.array(pvals)

    xys = [(uniq_v, np.mean(pvals <= uniq_v)) for uniq_v in np.unique(pvals)]

    area, prev_x, prev_y = 0, 0, 0
    for x, y in xys:
        area += (x - prev_x) * prev_y
        prev_x, prev_y = x, y

    area += (1 - prev_x) * prev_y
    return area


def KS_statistic(pvals: np.ndarray) -> float:
    """Kolmogorov-Smirnov test statistics"""
    D, _ = scipy.stats.kstest(pvals, 'uniform')
    return D
