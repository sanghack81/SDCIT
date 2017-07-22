import numpy as np

from sdcit.kcit import python_kcit
from sdcit.sdcit import SDCIT, mask_and_perm


def sim_hybrid(beta, gamma, n, alpha):
    ps = np.array([hybrid_check(beta, gamma) for _ in range(n)])
    temp = (ps <= alpha)
    return temp.mean()


def hybrid(X, Y, Z, Kx, Ky, Kz, Dz, beta=0.05, gamma=0.05):
    p1 = SDCIT(Kx, Ky, Kz, Dz)[1]
    if p1 <= beta:
        return p_transform(beta, gamma, p1)

    _, Pidx = mask_and_perm(Dz)
    p2 = python_kcit(X, Y[Pidx, :], Z)[2]
    if p2 <= gamma:
        return p_transform(beta, gamma, p1)

    return p_transform(beta, gamma, python_kcit(X, Y, Z)[2])


def hybrid_check(beta, gamma):
    p1 = np.random.rand()
    # 1
    if p1 <= beta:
        return p1
    # 1-beta
    p2 = np.random.rand()
    if p2 <= gamma:  # (1-beta)*gamma
        return p1
    # (1-beta) - (1-beta)*gamma
    # (1-beta)(1-gamma)
    return np.random.rand()


def xxxxxx(beta, gamma, alpha):
    mbmg = (1 - beta) * (1 - gamma)
    theta1 = alpha / (1 + mbmg)
    if theta1 <= beta:
        return theta1
    theta2 = (alpha - (1 - gamma) * beta) / (gamma + mbmg)
    return theta2


def p_transform(beta, gamma, hybrid_p):
    if hybrid_p <= beta:
        return hybrid_p * (1 + (1 - beta) * (1 - gamma))
    else:
        return beta * (1 + (1 - beta) * (1 - gamma)) + (hybrid_p - beta) * (gamma + (1 - beta) * (1 - gamma))


if __name__ == '__main__':
    # for _ in range(20):
    #     beta = np.random.rand()
    #     gamma = np.random.rand()
    #     alpha = np.random.rand()
    #     print(np.abs(sim_hybrid(beta, gamma, 2000000, alpha) - pred(beta, gamma, alpha)))

    for _ in range(20):
        beta = np.random.rand()
        gamma = np.random.rand()
        alpha = 0.05
        threshold = p_transform(beta, gamma, alpha)
        print((sim_hybrid(beta, gamma, 2000000, threshold) - alpha))
