import numpy as np
import pytest

from src.heston import HestonParams, simulate_heston_paths


def test_paths_are_finite_and_variance_nonnegative():
    s, v = simulate_heston_paths(100, HestonParams(2, .04, .6, -.7, .09), .03, 0, 1, 24, 1000, seed=4)
    assert s.shape == v.shape == (1000, 25)
    assert np.isfinite(s).all() and np.isfinite(v).all()
    assert (s > 0).all() and (v >= 0).all()


def test_invalid_correlation_rejected():
    with pytest.raises(ValueError):
        HestonParams(2, .04, .6, 1.1, .04).validate()


def test_antithetic_requires_even_paths():
    with pytest.raises(ValueError):
        simulate_heston_paths(100, HestonParams(2, .04, .6, 0, .04), .03, 0, 1, 10, 99)
