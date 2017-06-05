import numpy as np

from sdcit.hsic import HSIC
from sdcit.utils import rbf_kernel_with_median_heuristic, residual_kernel, residualize, columnwise_normalizes


def FCIT_K(KX, KY, KZ=None, eq_17_as_is=True, with_gp=True, sigma_squared=1e-3, seed=None, hsic_kws=None):
    """Flaxman et al. Residualization-based CI Test, X_||_Y | Z"""
    if seed is not None:
        np.random.seed(seed)
    if hsic_kws is None:
        hsic_kws = {}

    if KZ is None:
        return HSIC(KX, KY, **hsic_kws)

    RX_Z = residual_kernel(KX, KZ, eq_17_as_is=eq_17_as_is, with_gp=with_gp, sigma_squared=sigma_squared)
    RY_Z = residual_kernel(KY, KZ, eq_17_as_is=eq_17_as_is, with_gp=with_gp, sigma_squared=sigma_squared)

    return HSIC(RX_Z, RY_Z, **hsic_kws)


def FCIT(X, Y, Z=None, kern=rbf_kernel_with_median_heuristic, normalize=False, seed=None, hsic_kws=None):
    """Flaxman et al. Residualization-based CI Test, X_||_Y | Z"""
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
