"""Paired two-sided sign-flip permutation test over seeds (training and MAS reports)."""
from __future__ import annotations

import numpy as np

from config import N_PERMUTATIONS, SEED


def sign_flip_pvalue(differences, n_permutations: int = N_PERMUTATIONS, seed: int = SEED) -> float:
    """p-value of the mean paired difference, from random sign flips, with the plus-one
    correction. Each call uses its own generator, so the result depends only on the input."""
    diffs = np.asarray(differences, dtype=float)
    if np.allclose(diffs, 0):
        return 1.0
    signs = np.random.default_rng(seed).choice(np.asarray([-1.0, 1.0]), size=(n_permutations, len(diffs)))
    extreme = np.count_nonzero(np.abs((signs * diffs).mean(axis=1)) >= abs(float(diffs.mean())) - 1e-12)
    return float((extreme + 1) / (n_permutations + 1))
