"""Upper-tail Monte Carlo ranks, including degenerate null distributions."""

import numpy as np
import pytest

from sdcit.utils import p_value_of


@pytest.mark.parametrize(
    "statistic, null, expected",
    [
        (0, [0] * 100, 1),
        (10, list(range(1, 11)), 2 / 11),
        (11, list(range(1, 11)), 1 / 11),
        (-1, list(range(1, 11)), 1),
        (2, [3, 1, 2, 2], 4 / 5),
    ],
)
def test_upper_tail_includes_ties_and_observed_statistic(statistic, null, expected):
    assert p_value_of(statistic, null) == pytest.approx(expected)


def test_empty_null_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        p_value_of(0, [])


@pytest.mark.parametrize("pooled", [[0, 1, 2, 3, 4], [0, 0, 1, 3, 3], [0] * 5])
def test_exchangeable_ranks_are_superuniform(pooled):
    # Under exchangeability each pooled observation is equally likely to be
    # the observed statistic. Enumerate that exact finite reference experiment.
    # This checks the rank rule, not exchangeability of SDCIT's approximate null.
    pvalues = [
        p_value_of(value, pooled[:index] + pooled[index + 1:])
        for index, value in enumerate(pooled)
    ]
    for rank in range(len(pooled) + 1):
        alpha = rank / len(pooled)
        assert sum(p <= alpha for p in pvalues) / len(pooled) <= alpha


def test_random_upper_tails_match_direct_enumeration():
    rng = np.random.default_rng(173)
    for size in (1, 2, 7, 100):
        null = rng.integers(-3, 4, size=size).tolist()
        for statistic in range(-4, 5):
            expected = (1 + sum(value >= statistic for value in null)) / (size + 1)
            assert p_value_of(statistic, null) == expected


@pytest.mark.parametrize("implementation", ["SDCIT", "c_SDCIT"])
def test_constant_gram_matrices_do_not_reject(implementation):
    pytest.importorskip("sdcit.cython_impl.cy_sdcit")
    from sdcit import sdcit_mod

    kernel = np.ones((8, 8))
    statistic, pvalue, null = getattr(sdcit_mod, implementation)(
        kernel, kernel, kernel, seed=123, size_of_null_sample=100, with_null=True
    )
    assert statistic == 0
    np.testing.assert_array_equal(null, np.zeros(100))
    assert pvalue == 1
