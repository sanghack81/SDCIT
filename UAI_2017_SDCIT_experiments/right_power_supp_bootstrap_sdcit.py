import os

import numpy as np
from tqdm import trange

from UAI_2017_SDCIT_experiments.testing_utils import read_chaotic
from sdcit.sdcit import emp_MMSD, MMSD, perm_and_mask, penalized_distance
from sdcit.utils import p_value_of

if __name__ == '__main__':
    independent = 1
    gamma = 0.0
    N = 400
    if not os.path.exists('results/bootsdcit_mmd_p.csv'):
        with open('results/bootsdcit_mmd_p.csv', 'w') as f:
            for trial in trange(300):
                np.random.seed(trial)
                kx, ky, kz, Dz = read_chaotic(independent, gamma, trial, N)
                kxz = kx * kz
                bootmmd = emp_MMSD(kxz, ky, Dz, 100).mean()

                test_statistic, mask, Pidx = MMSD(kxz, ky, Dz)
                mask, Pidx = perm_and_mask(penalized_distance(Dz, mask))

                # avoid permutation between already permuted pairs.
                raw_null = emp_MMSD(kxz,
                                    ky[np.ix_(Pidx, Pidx)],
                                    penalized_distance(Dz, mask),
                                    1000)

                null = raw_null - raw_null.mean()

                print(bootmmd, p_value_of(bootmmd, null), sep=',', file=f, flush=True)

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv('results/bootsdcit_mmd_p.csv', names=['mmd', 'p'])
    sns.set()
    sns.distplot(df['p'])
    plt.savefig('figures/bootsdcit_dist.pdf')
    plt.close()
