import numpy as np

from sdcit.hsic import HSIC
from sdcit.kcit import residual_kernel_matrix_kernel_real
from sdcit.utils import rbf_kernel_with_median_heuristic, residual_kernel, residualize, residualize_real_and_kernel, columnwise_normalizes


def is_gram_matrix(X):
    n, m = X.shape
    return n == m and np.allclose(X, X.T)


def residual_kernel_auto(Y, X):
    """Residual or Residual Kernel matrix given """
    if X is None:
        return Y

    KX = KY = None
    if is_gram_matrix(X):
        KX, X = X, None
    if is_gram_matrix(Y):
        KY, Y = Y, None

    if KX is not None:
        if KY is not None:
            return residual_kernel(KY, KX)  # eigen-decomposed KY
        else:
            return residualize_real_and_kernel(Y, KX)  #
    else:
        if KY is not None:
            return residual_kernel_matrix_kernel_real(KY, X)  # KCIT-based, eigendecomposed KY
        else:
            return residualize(Y, X)  # typical Gaussian process based residuals


def FCIT_full_auto(X, Y, Z=None, Xcond=None, Ycond=None, kern=rbf_kernel_with_median_heuristic, hsic_kws={}):
    """X,Y, or Z can be either data or kernel matrix"""
    X = residual_kernel_auto(X, Xcond)
    Y = residual_kernel_auto(Y, Ycond)

    X = residual_kernel_auto(X, Z)
    Y = residual_kernel_auto(Y, Z)

    if not is_gram_matrix(X):
        X = kern(X)
    if not is_gram_matrix(Y):
        Y = kern(Y)

    return HSIC(X, Y, **hsic_kws)


def FCIT_K(KX, KY, KZ=None, eq_17_as_is=False, hsic_kws={}):
    """Flaxman et al. Residualization-based CI Test, X_||_Y | Z"""
    if KZ is None:
        return HSIC(KX, KY, **hsic_kws)

    RX_Z = residual_kernel(KX, KZ, eq_17_as_is=eq_17_as_is)
    RY_Z = residual_kernel(KY, KZ, eq_17_as_is=eq_17_as_is)

    return HSIC(RX_Z, RY_Z, **hsic_kws)


def FCIT(X, Y, Z=None, kern=rbf_kernel_with_median_heuristic, normalize=False, hsic_kw={}):
    """Flaxman et al. Residualization-based CI Test, X_||_Y | Z"""
    if normalize:
        X, Y, Z = columnwise_normalizes(X, Y, Z)

    if Z is None:
        return HSIC(kern(X), kern(Y), **hsic_kw)

    e_YZ = residualize(Y, Z)
    e_XZ = residualize(X, Z)

    if normalize:
        e_XZ, e_YZ = columnwise_normalizes(e_XZ, e_YZ)

    return HSIC(kern(e_XZ), kern(e_YZ), **hsic_kw)
