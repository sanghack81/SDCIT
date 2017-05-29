from sdcit.hsic import HSIC, HSIC_boot
from sdcit.utils import rbf_kernel_with_median_heuristic, residual_kernel, residualize, columnwise_normalize, columnwise_normalizes


def FCIT_K(KX, KY, KZ):
    RX_Z = residual_kernel(KX, KZ)
    RY_Z = residual_kernel(KY, KZ)

    # Approximation based HSIC seems not working very well under certain conditions...
    return HSIC_boot(RX_Z, RY_Z)


def FCIT(X, Y, Z=None):
    """Flaxman et al. Residualization-based CI Test"""
    if Z is not None:
        X, Y, Z = columnwise_normalizes(X, Y, Z)

        e_YZ = columnwise_normalize(residualize(Y, Z))
        e_XZ = columnwise_normalize(residualize(X, Z))

        if e_YZ.ndim == 1:
            e_YZ = e_YZ[:, None]
        if e_XZ.ndim == 1:
            e_XZ = e_XZ[:, None]

        return HSIC(*rbf_kernel_with_median_heuristic(e_XZ, e_YZ))
    else:
        return HSIC(*rbf_kernel_with_median_heuristic(X, Y))
