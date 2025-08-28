import numpy as np
from src.utils import normalize_scores, top_k_indices

def test_normalize_scores():
    a = np.array([1,2,3])
    n = normalize_scores(a)
    assert np.isclose(n[0], 0.0) and np.isclose(n[-1], 1.0)

def test_top_k_indices():
    a = np.array([1,5,3,4])
    idx = top_k_indices(a, 2)
    assert list(idx) == [1,3]
