import os
import warnings

import GPflow
import numpy as np
from numpy import eye, exp, sqrt, trace, diag, zeros
from scipy.stats import chi2, gamma
from sklearn.metrics import pairwise_distances

from sdcit.utils import centering, pdinv, reduced_eig, eigdec, columnwise_normalizes


def np2matlab(arr):
    import matlab.engine

    return matlab.double([[float(v) for v in row] for row in arr])


def kcit_original_matlab(x, y, z, seed=None, mateng=None, installed_at='~/Dropbox/research/2014 rcm/workspace/python/SDCIT/kcit'):
    return kcit(x, y, z, seed, mateng, installed_at)


def kcit(x, y, z, seed=None, mateng=None, installed_at='~/Dropbox/research/2014 rcm/workspace/python/SDCIT/kcit'):
    """Python-wrapper for KCIT by Zhang et al. (2011)"""
    import matlab.engine

    not_given = mateng is None
    try:
        if not_given:
            mateng = matlab.engine.start_matlab()
            dir_at = os.path.expanduser(installed_at)
            mateng.addpath(mateng.genpath(dir_at))

        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))
        statistic, v2, boot_p_value, v3, appr_p_value = mateng.CInd_test_new_withGP(np2matlab(x), np2matlab(y), np2matlab(z), 0.01, 0, nargout=5)
        return statistic, v2, boot_p_value, v3, appr_p_value
    finally:
        if not_given and mateng is not None:
            mateng.quit()


def kcit_lee(Kx, Ky, Kz, seed=None, mateng=None, installed_at='~/Dropbox/research/2014 rcm/workspace/python/SDCIT/kcit'):
    warnings.warn('KCIT without hyperparameter optimization')
    import matlab.engine

    not_given = mateng is None
    try:
        if not_given:
            mateng = matlab.engine.start_matlab()
            dir_at = os.path.expanduser(installed_at)
            mateng.addpath(mateng.genpath(dir_at))

        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))
        _, _, _, _, appr_p_value = mateng.CInd_test_new_withGP_Lee(Kx, Ky, Kz, 0.01, nargout=5)
        return appr_p_value
    finally:
        if not_given and mateng is not None:
            mateng.quit()


def gamcdf(t, a, b):
    return gamma.cdf(t, a, scale=b)


def gaminv(q, a, b):
    return gamma.ppf(q, a, scale=b)


def chi2rnd(df, m, n):
    return chi2.rvs(df, size=m * n).reshape((m, n))


def kernel(x, xKern=None, theta=0):
    if xKern is None:
        xKern = x
    n2 = pairwise_distances(x, xKern, metric='sqeuclidean')
    if theta == 0:
        theta = 2 / np.median(n2[np.tril(n2) > 0])

    wi2 = theta / 2
    kx = exp(-n2 * wi2)
    return kx




def residual_kernel_matrix_kernel_real(Kx, Z, num_eig):
    """K_X|Z"""
    assert len(Kx) == len(Z)
    assert num_eig <= len(Kx)
    T = len(Kx)
    D = Z.shape[1]
    I = eye(T)
    eig_Kx, eix = reduced_eig(*eigdec(Kx, num_eig))

    gp_x = GPflow.gpr.GPR(Z,
                          2 * sqrt(T) * eix @ diag(sqrt(eig_Kx)) / sqrt(eig_Kx[0]),
                          GPflow.kernels.RBF(D) + GPflow.kernels.White(D))
    gp_x.optimize()

    Kz_x = gp_x.kern.rbf.compute_K_symm(Z)

    P1_x = I - Kz_x @ pdinv(Kz_x + gp_x.kern.white.variance.value[0] * I)
    Kxz = P1_x @ Kx @ P1_x.T
    return Kxz


def python_kcit(x: np.ndarray, y: np.ndarray, z: np.ndarray, alpha=0.05, width=0.0, with_gp=False, noise=1e-3, num_bootstrap_for_null=5000):
    T = len(y)

    x, y, z = columnwise_normalizes(x, y, z)
    D = z.shape[1]

    width = width_heuristic_by_zhang(T, width)

    theta = 1 / (width ** 2 * D)
    Kx = centering(kernel(np.hstack([x, z / 2]), theta=theta))
    Ky = centering(kernel(y, theta=theta))

    if with_gp:
        Kxz = residual_kernel_matrix_kernel_real(Kx, z, min(400, T // 4))
        Kyz = residual_kernel_matrix_kernel_real(Ky, z, min(200, T // 5))
    else:
        Kz = centering(kernel(z, theta=theta))
        P1_x = P1_y = (eye(T) - Kz @ pdinv(Kz + noise * eye(T)))
        Kxz = P1_x @ Kx @ P1_x.T
        Kyz = P1_y @ Ky @ P1_y.T

    test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

    # null computation
    eig_Kxz, eivx = reduced_eig(*eigdec(Kxz))
    eig_Kyz, eivy = reduced_eig(*eigdec(Kyz))

    eiv_prodx = eivx @ diag(sqrt(eig_Kxz))
    eiv_prody = eivy @ diag(sqrt(eig_Kyz))

    num_eigx = eiv_prodx.shape[1]
    num_eigy = eiv_prody.shape[1]
    size_u = num_eigx * num_eigy
    uu = zeros((T, size_u))
    for i in range(num_eigx):
        for j in range(num_eigy):
            uu[:, i * num_eigy + j] = eiv_prodx[:, i] * eiv_prody[:, j]

    uu_prod = uu @ uu.T if size_u > T else uu.T @ uu
    eig_uu = reduced_eig(eigdec(uu_prod, min(T, size_u))[0])

    boot_critical_val, boot_p_val = null_by_bootstrap(test_statistic, num_bootstrap_for_null, alpha, eig_uu[:, None])
    appr_critical_val, appr_p_val = null_by_gamma_approx(test_statistic, alpha, uu_prod)

    return test_statistic, boot_critical_val, boot_p_val, appr_critical_val, appr_p_val


def width_heuristic_by_zhang(T, width):
    if width == 0:
        if T <= 200:
            width = 1.2
        elif T < 1200:
            width = 0.7
        else:
            width = 0.4
    return width


def null_by_gamma_approx(Sta, alpha, uu_prod):
    """critical value and p-value based on Gamma distribution approximation"""
    mean_appr = trace(uu_prod)
    var_appr = 2 * trace(uu_prod ** 2)
    k_appr = mean_appr ** 2 / var_appr
    theta_appr = var_appr / mean_appr
    Cri_appr = gaminv(1 - alpha, k_appr, theta_appr)
    p_appr = 1 - gamcdf(Sta, k_appr, theta_appr)
    return Cri_appr, p_appr


def null_by_bootstrap(Sta, T_BS, alpha, eig_uu):
    """critical value and p-value based on bootstrapping"""
    null_dstr = eig_uu.T @ chi2rnd(1, len(eig_uu), T_BS)
    null_dstr = np.sort(null_dstr.squeeze())

    critical_val = null_dstr[int((1 - alpha) * T_BS)]
    p_val = np.sum(null_dstr > Sta) / T_BS
    return critical_val, p_val
