import os

import numpy as np
from GPflow.gpr import GPR
from GPflow.kernels import RBF, White
from numpy import eye, sqrt, trace, diag, zeros
from scipy.stats import chi2, gamma

from sdcit.utils import centering, pdinv, truncated_eigen, eigdec, columnwise_normalizes, residual_kernel, rbf_kernel_median


def np2matlab(arr: np.ndarray):
    import matlab.engine

    return matlab.double([[float(v) for v in row] for row in arr])


def matlab_kcit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, seed: int = None, mateng=None, installed_at=None):
    """Python-wrapper for KCIT by Zhang et al. (2011)


    References
    ----------
    Zhang, K., Peters, J., Janzing, D., & Schölkopf, B. (2011). Kernel-based Conditional Independence Test and Application in Causal Discovery.
    In Proceedings of the 27th Conference on Uncertainty in Artificial Intelligence (pp. 804–813). Corvallis, Oregon: AUAI Press.
    """
    import matlab.engine

    not_given = mateng is None
    try:
        if not_given:
            mateng = matlab.engine.start_matlab()
            dir_at = os.path.expanduser(installed_at)
            mateng.addpath(mateng.genpath(dir_at))

        if seed is not None:
            mateng.RandStream.setGlobalStream(mateng.RandStream('mcg16807', 'Seed', seed))

        statistic, v2, boot_p_value, v3, appr_p_value = mateng.CInd_test_new_withGP(np2matlab(X), np2matlab(Y), np2matlab(Z), 0.01, 0, nargout=5)
        return statistic, v2, boot_p_value, v3, appr_p_value
    finally:
        if not_given and mateng is not None:
            mateng.quit()


def gamcdf(t: float, shape: float, scale: float) -> float:
    """Cumulative distribution function of Gamma distribution"""
    return gamma.cdf(t, shape, scale=scale)


def gaminv(q: float, shape: float, scale: float) -> float:
    return gamma.ppf(q, shape, scale=scale)


def chi2rnd(df, m, n):
    return chi2.rvs(df, size=m * n).reshape((m, n))


def residual_kernel_matrix_kernel_real(Kx, Z, num_eig):
    """K_X|Z"""
    assert len(Kx) == len(Z)
    assert num_eig <= len(Kx)

    T = len(Kx)
    D = Z.shape[1]
    I = eye(T)
    eig_Kx, eix = truncated_eigen(*eigdec(Kx, num_eig))

    gp_model = GPR(Z, 2 * sqrt(T) * eix @ diag(sqrt(eig_Kx)) / sqrt(eig_Kx[0]), RBF(D, ARD=True) + White(D))
    gp_model.optimize()

    Kz_x = gp_model.kern.rbf.compute_K_symm(Z)
    sigma_squared = gp_model.kern.white.variance.value[0]

    P = I - Kz_x @ pdinv(Kz_x + sigma_squared * I)
    return P @ Kx @ P.T


def python_kcit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, alpha=0.05, with_gp=True, noise=1e-3, num_bootstrap_for_null=5000, normalize=False, kern=rbf_kernel_median, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T = len(Y)

    if normalize:
        X, Y, Z = columnwise_normalizes(X, Y, Z)

    Kx = centering(kern(np.hstack([X, Z])))  # originally [x, z /2]
    Ky = centering(kern(Y))

    if with_gp:
        Kxz = residual_kernel_matrix_kernel_real(Kx, Z, min(200, T // 5))  # originally, min(400, T // 4)
        Kyz = residual_kernel_matrix_kernel_real(Ky, Z, min(200, T // 5))
    else:
        Kz = centering(kern(Z))
        P1 = eye(T) - Kz @ pdinv(Kz + noise * eye(T))  # pdinv(I+K/noise)
        Kxz = P1 @ Kx @ P1.T
        Kyz = P1 @ Ky @ P1.T

    test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

    # null computation
    return kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic)


def python_kcit_K(Kx: np.ndarray, Ky: np.ndarray, Kz: np.ndarray, alpha=0.05, with_gp=True, sigma_squared=1e-3, num_bootstrap_for_null=5000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T = len(Kx)

    Kx, Ky, Kz = centering(Kx * Kz), centering(Ky), centering(Kz)

    if with_gp:
        Kxz = residual_kernel(Kx, Kz, use_expectation=False, with_gp=with_gp)
        Kyz = residual_kernel(Ky, Kz, use_expectation=False, with_gp=with_gp)
    else:
        P = eye(T) - Kz @ pdinv(Kz + sigma_squared * eye(T))
        Kxz = P @ Kx @ P.T
        Kyz = P @ Ky @ P.T

    test_statistic = (Kxz * Kyz).sum()  # trace(Kxz @ Kyz)

    return kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic)


def kcit_null(Kxz, Kyz, T, alpha, num_bootstrap_for_null, test_statistic):
    # null computation
    eig_Kxz, eivx = truncated_eigen(*eigdec(Kxz))
    eig_Kyz, eivy = truncated_eigen(*eigdec(Kyz))

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
    eig_uu = truncated_eigen(eigdec(uu_prod, min(T, size_u))[0])

    boot_critical_val, boot_p_val = _null_by_bootstrap(test_statistic, num_bootstrap_for_null, alpha, eig_uu[:, None])
    appr_critical_val, appr_p_val = _null_by_gamma_approx(test_statistic, alpha, uu_prod)

    return test_statistic, boot_critical_val, boot_p_val, appr_critical_val, appr_p_val


def _null_by_gamma_approx(statistic, alpha, uu_prod):
    """critical value and p-value based on Gamma distribution approximation"""
    mean_appr = trace(uu_prod)
    var_appr = 2 * trace(uu_prod ** 2)

    k_appr = mean_appr ** 2 / var_appr  # type: float
    theta_appr = var_appr / mean_appr  # type: float

    critical_value = gaminv(1 - alpha, k_appr, theta_appr)
    p_value = 1 - gamcdf(statistic, k_appr, theta_appr)

    return critical_value, p_value


def _null_by_bootstrap(statistic, num_bootstrap, alpha, eig_uu):
    """critical value and p-value based on bootstrapping"""
    null_dstr = eig_uu.T @ chi2rnd(1, len(eig_uu), num_bootstrap)
    null_dstr = np.sort(null_dstr.squeeze())

    critical_value = null_dstr[int((1 - alpha) * num_bootstrap)]
    p_value = np.sum(null_dstr > statistic) / num_bootstrap

    return critical_value, p_value
