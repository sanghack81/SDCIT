import math

import numpy as np
import scipy.stats

from sdcit.flaxman import FCIT, FCIT_K
from sdcit.kcit import python_kcit_K, python_kcit
from sdcit.sdcit import SDCIT
from sdcit.utils import rbf_kernel_median, columnwise_normalizes, residualize


def data(n):
    np.random.seed(0)
    Z = np.sort(np.random.rand(n)) * math.pi

    Y = scipy.stats.gamma.rvs(1, scale=2, size=n) - 2  # centered at 0
    X = scipy.stats.gamma.rvs(1, scale=2, size=n) - 2  # centered at 0
    Y *= np.cos(math.pi / 2 - Z)
    X *= np.cos(math.pi / 2 - Z)

    X, Y, Z = X[:, None], Y[:, None], Z[:, None]
    X, Y, Z = columnwise_normalizes(X, Y, Z)

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set()
    plt.scatter(Z, X)
    plt.scatter(Z, Y)
    plt.savefig('figures/Z_XY.pdf')
    plt.close()

    RX = residualize(X, Z)
    RY = residualize(Y, Z)
    plt.scatter(Z, RX)
    plt.scatter(Z, RY)
    plt.savefig('figures/Z_RXRY.pdf')
    plt.close()

    plt.scatter(RX, RY)
    plt.savefig('figures/RXRY.pdf')
    plt.close()

    # print(HSIC(rbf_kernel_with_median_heuristic(RX), rbf_kernel_with_median_heuristic(RY)))
    print('fcit', FCIT(X, Y, Z))
    print('fcit_K', FCIT_K(*rbf_kernel_median(X, Y, Z)))
    # print('fcit_K', FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, sigma_squared=1))
    # print('fcit_K', FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, sigma_squared=0.1))
    # print('fcit_K', FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, sigma_squared=0.01))
    # print('fcit_K', FCIT_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, sigma_squared=0.001))
    print('sdcit', SDCIT(*rbf_kernel_median(X, Y, Z))[1])
    print('kcit', python_kcit(X, Y, Z)[2])
    print('kcit_K', python_kcit_K(*rbf_kernel_median(X, Y, Z))[2])
    # print('kcit_K', python_kcit_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, noise=1)[2])
    # print('kcit_K', python_kcit_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, noise=0.01)[2])
    # print('kcit_K', python_kcit_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, noise=0.0001)[2])
    # print('kcit_K', python_kcit_K(*rbf_kernel_with_median_heuristic(X, Y, Z), with_gp=False, noise=0.000001)[2])


if __name__ == '__main__':
    data(400)
