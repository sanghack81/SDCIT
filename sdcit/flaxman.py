import numpy as np

from sdcit.hsic import HSIC
from sdcit.utils import rbf_kernel_median, residual_kernel, residualize, columnwise_normalizes


def FCIT_noniid_K(Kx, Ky, cond_Kx, cond_Ky, Kz=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    RX1 = residual_kernel(Kx, cond_Kx)
    RY1 = residual_kernel(Ky, cond_Ky)

    return FCIT_K(RX1, RY1, Kz)


def FCIT_noniid(X, Y, Cond_X, Cond_Y, Z=None, seed=None):
    """Flaxman et al. Residualization-based CI Test, X_||_Y | Z

    References
    ----------
    Flaxman, S. R., Neill, D. B., & Smola, A. J. (2016). Gaussian Processes for Independence Tests with Non-iid Data in Causal Inference.
    ACM Transactions on Intelligent Systems and Technology, 7(2), 1–23.
    """
    if seed is not None:
        np.random.seed(seed)

    RX1 = residualize(X, Cond_X)
    RY1 = residualize(Y, Cond_Y)

    return FCIT(RX1, RY1, Z)


def FCIT_K(Kx, Ky, Kz=None, use_expectation=True, with_gp=True, sigma_squared=1e-3, seed=None, hsic_kws=None):
    if seed is not None:
        np.random.seed(seed)

    if hsic_kws is None:
        hsic_kws = {}

    if Kz is None:
        return HSIC(Kx, Ky, **hsic_kws)

    RX_Z = residual_kernel(Kx, Kz, use_expectation=use_expectation, with_gp=with_gp, sigma_squared=sigma_squared)
    RY_Z = residual_kernel(Ky, Kz, use_expectation=use_expectation, with_gp=with_gp, sigma_squared=sigma_squared)

    return HSIC(RX_Z, RY_Z, **hsic_kws)


def FCIT(X, Y, Z=None, kern=rbf_kernel_median, normalize=False, seed=None, hsic_kws=None):
    """Flaxman et al. Residualization-based CI Test, X_||_Y | Z

    References
    ----------
    Flaxman, S. R., Neill, D. B., & Smola, A. J. (2016). Gaussian Processes for Independence Tests with Non-iid Data in Causal Inference.
    ACM Transactions on Intelligent Systems and Technology, 7(2), 1–23.
    """
    if seed is not None:
        np.random.seed(seed)

    if hsic_kws is None:
        hsic_kws = {}

    if normalize:
        X, Y, Z = columnwise_normalizes(X, Y, Z)

    if Z is None:
        return HSIC(kern(X), kern(Y), **hsic_kws)

    e_YZ = residualize(Y, Z)
    e_XZ = residualize(X, Z)

    if normalize:
        e_XZ, e_YZ = columnwise_normalizes(e_XZ, e_YZ)

    return HSIC(kern(e_XZ), kern(e_YZ), **hsic_kws)
