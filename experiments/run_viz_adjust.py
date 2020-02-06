import numpy as np
from sklearn.linear_model import LinearRegression

from experiments.exp_utils import read_chaotic
from sdcit.sdcit_mod import mask_and_perm, MMSD, adjust_errors, emp_MMSD, penalized_distance
from sdcit.utils import K2D, p_value_of
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def viz_adjust_errors(null_errors, null, error=None, test_statistic=None):
    if error is not None:
        assert test_statistic is not None

    model = LinearRegression().fit(null_errors[:, None], null[:, None])
    beta = max(0, model.coef_[0, 0])

    sns.set(style='white', font_scale=1.5)
    plt.figure()
    x = pd.Series(1000 * null_errors, name="error distribution x1000")
    y = pd.Series(1000 * null, name="null distribution x1000")
    graph = sns.jointplot(x=x, y=y, scatter_kws={'alpha': 0.3}, line_kws={'color': 'r'}, stat_func=None, kind='reg')
    graph.y = [test_statistic]
    graph.x = [error]
    graph.plot_joint(plt.scatter, marker='x', c='r', s=50)

    # plt.title('Before adjust')
    sns.despine()
    plt.savefig('results/viz_adjust_error.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    if error is not None:
        return null - null_errors * beta, test_statistic - error * beta
    else:
        return null - null_errors * beta


def viz_SDCIT_adjust(Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray, Dz=None, size_of_null_sample=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if Dz is None:
        Dz = K2D(Kz)

    Kxz = Kx * Kz

    test_statistic, error_statistic, mask, _ = MMSD(Ky, Kz, Kxz, Dz)
    mask, Pidx = mask_and_perm(penalized_distance(Dz, mask))

    # avoid permutation between already permuted pairs.
    mmsd_distr_under_null, error_distr_under_null = emp_MMSD(Kxz, Ky[np.ix_(Pidx, Pidx)], Kz,
                                                             penalized_distance(Dz, mask), size_of_null_sample)

    fix_null, fix_test_statistic = viz_adjust_errors(error_distr_under_null, mmsd_distr_under_null, error_statistic,
                                                     test_statistic)

    sns.set(style='white', font_scale=1)
    plt.figure(figsize=[4, 1.5])
    g = sns.distplot(mmsd_distr_under_null * 1000, hist=True, kde=False, label='unadjusted')
    g.set(xticklabels=[])
    g.set(yticklabels=[])
    g = sns.distplot(fix_null * 1000, hist=True, kde=False, label='adjusted')
    g.set(yticklabels=[])
    g.set(xticklabels=[])
    plt.legend()
    sns.despine()
    plt.savefig('results/viz_adjust_error_two_dists.pdf', transparent=True, bbox_inches='tight', pad_inches=0.02)
    plt.close()


if __name__ == '__main__':
    kx, ky, kz, Dz = read_chaotic(1, 0.0, 0, 200)
    viz_SDCIT_adjust(kx, ky, kz, seed=0)
