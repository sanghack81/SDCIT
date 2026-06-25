import numpy as np
import pytest

from sdcit.synthetic_data import (
    henon,
    normalize,
    zhang2012,
    symmetric_zhang2012
)

def test_normalize():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    norm_X = normalize(X)
    
    # Means should be 0, stds should be 1
    assert np.allclose(np.mean(norm_X, axis=0), 0)
    assert np.allclose(np.std(norm_X, axis=0), 1)

def test_henon():
    n = 100
    noise_dim = 2
    
    # Independent case
    X, Y, Z = henon(seed=42, n=n, gamma=0.1, independence=True, noise_dim=noise_dim)
    assert X.shape == (n, 2 + noise_dim)
    assert Y.shape == (n, 2 + noise_dim)
    assert Z.shape == (n, 2)
    
    # Dependent case
    X2, Y2, Z2 = henon(seed=42, n=n, gamma=0.1, independence=False, noise_dim=noise_dim)
    assert X2.shape == (n, 2 + noise_dim)
    assert Y2.shape == (n, 2 + noise_dim)
    assert Z2.shape == (n, 2)

@pytest.mark.parametrize("dimensions", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("the_case", [1, 2])
@pytest.mark.parametrize("independent", [True, False])
def test_zhang2012(dimensions, the_case, independent):
    N = 50
    X, Y, Z = zhang2012(seed=42, N=N, dimensions=dimensions, the_case=the_case, independent=independent)
    
    assert X.shape == (N, 1)
    assert Y.shape == (N, 1)
    
    if the_case == 1:
        # For case 1, Z has 'dimensions' size
        assert Z.shape == (N, dimensions)
    else:
        # For case 2, Z also has 'dimensions' size up to 5
        assert Z.shape == (N, dimensions)

@pytest.mark.parametrize("dimensions", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("the_case", [1, 2])
@pytest.mark.parametrize("independent", [True, False])
def test_symmetric_zhang2012(dimensions, the_case, independent):
    N = 30
    X, Y, Z = symmetric_zhang2012(seed=42, N=N, dimensions=dimensions, the_case=the_case, independent=independent)
    
    assert X.shape == (N, 1)
    assert Y.shape == (N, 1)
    assert Z.shape == (N, dimensions)

def test_zhang2012_invalid_dimension():
    with pytest.raises(Exception, match='Between 1 and 5 dimensions supported'):
        zhang2012(seed=42, N=10, dimensions=6, the_case=2)
