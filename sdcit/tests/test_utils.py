import numpy as np
from sdcit.utils import (
    ensure_symmetric,
    centering,
    columnwise_normalize,
    columnwise_normalizes,
    truncated_eigen,
    eigdec,
    pdinv,
    p_value_of,
    rbf_kernel_median,
    random_seeds,
    K2D
)

def test_ensure_symmetric():
    # Asymmetric matrix
    M = np.array([[1.0, 2.0], [4.0, 5.0]])
    sym = ensure_symmetric(M)
    assert np.allclose(sym, sym.T)
    assert np.allclose(sym, [[1.0, 3.0], [3.0, 5.0]])

def test_centering():
    # None handling
    assert centering(None) is None
    
    # 3x3 matrix handling
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    centered = centering(M)
    # The sum of rows and columns of a centered matrix should be close to 0
    assert np.allclose(centered.sum(axis=0), 0)
    assert np.allclose(centered.sum(axis=1), 0)

def test_columnwise_normalize():
    assert columnwise_normalize(None) is None
    
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    norm_X = columnwise_normalize(X)
    
    # Means should be 0, stds should be 1
    assert np.allclose(np.mean(norm_X, axis=0), 0)
    assert np.allclose(np.std(norm_X, axis=0), 1)

def test_columnwise_normalizes():
    X1 = np.array([[1.0], [3.0]])
    X2 = np.array([[2.0], [4.0]])
    res = columnwise_normalizes(X1, X2)
    assert len(res) == 2
    assert np.allclose(np.mean(res[0], axis=0), 0)
    assert np.allclose(np.std(res[0], axis=0), 1)

def test_pdinv():
    # Positive definite matrix
    A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    A_inv = pdinv(A)
    # A * A^-1 should be identity
    assert np.allclose(A @ A_inv, np.eye(3))

def test_truncated_eigen():
    vals = np.array([100.0, 10.0, 1.0, 0.0000001])
    vecs = np.eye(4)
    # The default threshold is 1e-5 relative to max
    t_vals, t_vecs = truncated_eigen(vals, vecs)
    
    # Should keep the first 3 (100 is max, 100 * 1e-5 = 0.001)
    # 0.0000001 < 0.001 so it drops.
    assert len(t_vals) == 3
    assert t_vecs.shape == (4, 3)

    # Without vecs
    t_vals_only = truncated_eigen(vals)
    assert len(t_vals_only) == 3

def test_eigdec():
    # Symmetric matrix
    K = np.array([[2.0, 1.0], [1.0, 2.0]])
    w, v = eigdec(K)
    # Check if descending order
    assert w[0] >= w[1]
    
    # K v = \lambda v
    assert np.allclose(K @ v[:, 0], w[0] * v[:, 0])
    
def test_p_value_of():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    p = p_value_of(9.5, data)
    assert np.isclose(p, 2 / 11)
    
    p2 = p_value_of(11, data)
    assert np.isclose(p2, 1 / 11)

def test_rbf_kernel_median():
    X = np.random.randn(10, 2)
    K = rbf_kernel_median(X)
    assert K.shape == (10, 10)
    
    # Test with multiple
    Y = np.random.randn(10, 3)
    K_list = rbf_kernel_median(X, Y)
    assert len(K_list) == 2
    assert K_list[0].shape == (10, 10)
    assert K_list[1].shape == (10, 10)

def test_random_seeds():
    seed = random_seeds()
    assert isinstance(seed, int)
    
    seeds = random_seeds(n=5)
    assert isinstance(seeds, list)
    assert len(seeds) == 5
    assert len(set(seeds)) > 1 # very likely distinct

def test_K2D():
    assert K2D(None) is None
    
    # Simple diagonal kernel -> Distance should be sqrt(2) everywhere except 0 on diag
    K = np.eye(3)
    D = K2D(K)
    
    expected_D = np.array([
        [0.0, np.sqrt(2), np.sqrt(2)],
        [np.sqrt(2), 0.0, np.sqrt(2)],
        [np.sqrt(2), np.sqrt(2), 0.0]
    ])
    assert np.allclose(D, expected_D)
