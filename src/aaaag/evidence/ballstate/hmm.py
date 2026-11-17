"""Two-state (alive, dead) forward-backward, vectorised as a parallel prefix scan.

The forward recursion alpha_t = (alpha_{t-1} A) * e_t is a product of 2x2 matrices
M_t = A diag(e_t), and matrix products are associative, so every prefix (and suffix) product is
computed in ceil(log2 T) vectorised passes (Hillis-Steele) instead of a Python loop over frames.
Everything stays in log space, so long runs of strong evidence cannot underflow.

Non-causal by construction: each frame's posterior uses the whole period, so there is no lag.
"""

from __future__ import annotations

import numpy as np

ALIVE, DEAD = 0, 1


def _log_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Batched (n, 2, 2) log-space matrix product: out[i, j] = logsumexp_k left[i, k] + right[k, j]."""
    pair = left[:, :, :, None] + right[:, None, :, :]  # (n, i, k, j)
    out = np.logaddexp(pair[:, :, 0, :], pair[:, :, 1, :])
    return out - out.max(axis=(1, 2), keepdims=True)  # rescale: posteriors only need ratios


def _scan(mats: np.ndarray, reverse: bool) -> np.ndarray:
    """Inclusive prefix products M_1..M_t (or suffix products M_t..M_T with reverse=True)."""
    out = mats.copy()
    n, step = len(out), 1
    while step < n:
        if reverse:
            out[:-step] = _log_matmul(out[:-step], out[step:])
        else:
            out[step:] = _log_matmul(out[:-step], out[step:])
        step *= 2
    return out


def posterior_dead(
    llr: np.ndarray, *, p_alive_to_dead: float, p_dead_to_alive: float, prior_dead: float
) -> np.ndarray:
    """P(dead | all evidence) per frame for one contiguous span.

    llr: (n,) per-frame log P(obs | dead) - log P(obs | alive); 0 (or NaN) means no evidence.
    The transition probabilities are per frame, i.e. 1 / (mean state duration in frames), and
    `prior_dead` is the state before the first frame.
    """
    llr = np.nan_to_num(np.asarray(llr, dtype=np.float64), nan=0.0)
    n = len(llr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    log_a = np.log(
        np.array(
            [[1.0 - p_alive_to_dead, p_alive_to_dead], [p_dead_to_alive, 1.0 - p_dead_to_alive]]
        )
    )
    emit = np.zeros((n, 2))
    emit[:, DEAD] = llr
    mats = log_a[None, :, :] + emit[:, None, :]  # M_t[i, j] = log A[i, j] + log e_t(j)

    start = np.log(np.array([1.0 - prior_dead, prior_dead]))
    prefix = _scan(mats, reverse=False)
    alpha = np.logaddexp(start[0] + prefix[:, 0, :], start[1] + prefix[:, 1, :])  # (n, 2)
    suffix = _scan(mats, reverse=True)
    beta = np.zeros((n, 2))
    beta[:-1] = np.logaddexp(suffix[1:, :, 0], suffix[1:, :, 1])  # M_{t+1}..M_n times ones
    gamma = alpha + beta
    return 1.0 / (1.0 + np.exp(np.clip(gamma[:, ALIVE] - gamma[:, DEAD], -700.0, 700.0)))


def posterior_dead_reference(
    llr: np.ndarray, *, p_alive_to_dead: float, p_dead_to_alive: float, prior_dead: float
) -> np.ndarray:
    """The textbook scaled forward-backward loop. Slow; tests check posterior_dead against it."""
    llr = np.nan_to_num(np.asarray(llr, dtype=np.float64), nan=0.0)
    n = len(llr)
    a = np.array(
        [[1.0 - p_alive_to_dead, p_alive_to_dead], [p_dead_to_alive, 1.0 - p_dead_to_alive]]
    )
    emit = np.stack([np.ones(n), np.exp(np.clip(llr, -50, 50))], axis=1)
    alpha = np.zeros((n, 2))
    state = np.array([1.0 - prior_dead, prior_dead])
    for t in range(n):
        state = (state @ a) * emit[t]
        state /= state.sum()
        alpha[t] = state
    beta = np.ones((n, 2))
    for t in range(n - 2, -1, -1):
        beta[t] = a @ (emit[t + 1] * beta[t + 1])
        beta[t] /= beta[t].sum()
    gamma = alpha * beta
    return gamma[:, DEAD] / gamma.sum(axis=1)
