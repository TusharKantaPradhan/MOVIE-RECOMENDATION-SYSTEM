from typing import List, Tuple
import numpy as np
import pandas as pd
from .utils import normalize_scores

class HybridRecommender:
    """
    Combine CF and CBF scores via weighted sum.
    """
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def combine(self, cf_recs: List[Tuple[int,float]], cbf_recs: List[Tuple[int,float]], top_k: int = 10):
        # unify movie id space
        from collections import defaultdict
        scores_cf = defaultdict(float)
        scores_cbf = defaultdict(float)

        for mid, s in cf_recs:
            scores_cf[int(mid)] = s
        for mid, s in cbf_recs:
            scores_cbf[int(mid)] = s

        all_ids = sorted(set(scores_cf.keys()) | set(scores_cbf.keys()))
        cf = np.array([scores_cf.get(i, 0.0) for i in all_ids])
        cb = np.array([scores_cbf.get(i, 0.0) for i in all_ids])

        cf = normalize_scores(cf)
        cb = normalize_scores(cb)
        final = self.alpha*cf + (1-self.alpha)*cb
        order = np.argsort(final)[::-1][:top_k]
        return [(int(all_ids[i]), float(final[i])) for i in order]
