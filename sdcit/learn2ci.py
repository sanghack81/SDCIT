import numpy as np

from sdcit.hsic import HSIC_boot
from sdcit.kcit import python_kcit_K
from sdcit.sdcit import SDCIT, mask_and_perm
from sdcit.utils import K2D


def ci_test(KX, KY, KZ=None, alpha=0.05):
    if KZ is None:
        return HSIC_boot(KX, KY)

    kcit_p_value = python_kcit_K(KX, KY, KZ)[2]

    # kcit says null.
    if kcit_p_value > alpha:
        return kcit_p_value

    DZ = K2D(KZ)
    # is kernel choice reasonable?
    mask, perm = mask_and_perm(DZ)
    PKY = KY[np.ix_(perm, perm)]
    null_kcit_p_value = python_kcit_K(KX, PKY, KZ)[2]
    # there seems no problem with kernel choice
    if null_kcit_p_value > alpha:
        return kcit_p_value

    # problematic?
    return SDCIT(KX, KY, KZ, DZ)[1]
