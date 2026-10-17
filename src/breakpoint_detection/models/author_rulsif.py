"""RuLSIF author algorithm; TS-CP2's fixed lambda overrides source search.

Liu et al., 2013: https://www.ms.k.u-tokyo.ac.jp/sugi/2013/CDRuLSIF.pdf
Source commit: 496042e79b33c794cf2d6807496499763818508a.
Original MIT notice accompanies this file. MATLAB randperm is not emulated.
"""
import numpy as np

N = 20
ALPHA = 0.1  # User-selected historical branch; target-paper Yahoo alpha is unconfirmed.
REGULARIZATION = 0.01
FOLDS = 5
MULTIPLIERS = np.array([.6, .8, 1., 1.2, 1.4])


def squared_distances(a, b):
    # Direct differences avoid cancellation in the equivalent norm expansion.
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def fit_direction(numerator, denominator, *, alpha=ALPHA):
    rng = np.random.RandomState(0)
    centers = numerator[rng.permutation(len(numerator))[:min(100, len(numerator))]]
    all_samples = np.concatenate((denominator, numerator))
    squared = squared_distances(all_samples, all_samples)
    upper = squared[np.triu_indices(len(all_samples), 1)]
    positive = upper[upper > 0]
    if not len(positive):
        raise ValueError("Author median bandwidth is undefined for a constant buffer")
    sigmas = np.sqrt(.5 * np.median(positive)) * MULTIPLIERS
    nu = np.exp(-squared_distances(numerator, centers)[None, :, :] / (2 * sigmas[:, None, None] ** 2))
    de = np.exp(-squared_distances(denominator, centers)[None, :, :] / (2 * sigmas[:, None, None] ** 2))
    nu_orders, de_orders = [], []
    for _ in sigmas:
        nu_orders.append(rng.permutation(len(numerator)))
        de_orders.append(rng.permutation(len(denominator)))
    nu = np.take_along_axis(nu, np.array(nu_orders)[:, :, None], axis=1)
    de = np.take_along_axis(de, np.array(de_orders)[:, :, None], axis=1)
    nu_folds = np.floor(np.arange(len(numerator)) * FOLDS / len(numerator)).astype(int)
    de_folds = np.floor(np.arange(len(denominator)) * FOLDS / len(denominator)).astype(int)
    train_nu = np.stack([nu[:, nu_folds != f] for f in range(FOLDS)], axis=1)
    train_de = np.stack([de[:, de_folds != f] for f in range(FOLDS)], axis=1)
    held_nu = np.stack([nu[:, nu_folds == f] for f in range(FOLDS)], axis=1)
    held_de = np.stack([de[:, de_folds == f] for f in range(FOLDS)], axis=1)
    H = (alpha * (train_nu.swapaxes(-1, -2) @ train_nu) / train_nu.shape[2]
         + (1-alpha) * (train_de.swapaxes(-1, -2) @ train_de) / train_de.shape[2])
    theta = np.linalg.solve(H + REGULARIZATION * np.eye(len(centers)), train_nu.mean(axis=2)[..., None])
    g_nu, g_de = (held_nu @ theta)[..., 0], (held_de @ theta)[..., 0]
    risks = np.mean(alpha * np.mean(g_nu**2, axis=2) / 2
                    + (1-alpha) * np.mean(g_de**2, axis=2) / 2
                    - np.mean(g_nu, axis=2), axis=1)
    best = int(np.argmin(risks))
    sigma = float(sigmas[best])
    # Full sample order is immaterial once sigma is fixed; retain source centers.
    phi_nu = np.exp(-squared_distances(numerator, centers) / (2 * sigma**2))
    phi_de = np.exp(-squared_distances(denominator, centers) / (2 * sigma**2))
    H = (alpha * phi_nu.T @ phi_nu / len(numerator)
         + (1-alpha) * phi_de.T @ phi_de / len(denominator))
    theta_final = np.linalg.solve(H + REGULARIZATION * np.eye(len(centers)), phi_nu.mean(axis=0))
    g_nu, g_de = phi_nu @ theta_final, phi_de @ theta_final
    pe = np.mean(g_nu) - .5 * (alpha * np.mean(g_nu**2) + (1-alpha) * np.mean(g_de**2)) - .5
    return float(pe), sigma, risks


def paired_samples(values, start, embedding, n=N):
    vectors = np.lib.stride_tricks.sliding_window_view(values, embedding)
    paired = vectors[start:start+2*n].copy()
    if paired.shape != (2*n, embedding):
        raise ValueError("Incomplete author buffer")
    scale = paired.std(axis=0, ddof=1)
    if np.any(scale == 0):
        raise ValueError("Source normalization has a zero coordinate std")
    paired /= scale
    return paired[:n], paired[n:]


def one_direction(values, embedding, n=N, alpha=ALPHA):
    count = len(values) - (2*n+embedding-1) + 1
    scores = np.empty(count)
    sigmas = np.empty(count)
    risks = np.empty((count, len(MULTIPLIERS)))
    for start in range(count):
        before, after = paired_samples(values, start, embedding, n)
        scores[start], sigmas[start], risks[start] = fit_direction(after, before, alpha=alpha)
    return scores, sigmas, risks


def score_series(values, embedding, *, n=N, alpha=ALPHA):
    """Execute two entire-series scans, as demo.m, and reverse compact output.

    Scoring step is one. Event anchoring is separate from demo's plot padding.
    Both boundary positions and literal plotting positions are returned.
    """
    if n < FOLDS or n % FOLDS or not 0 <= alpha < 1:
        raise ValueError('n must be a positive multiple of five; alpha must be in [0,1)')
    forward, sigma_f, risk_f = one_direction(values, embedding, n, alpha)
    backward, sigma_b, risk_b = one_direction(values[::-1], embedding, n, alpha)
    compact = forward + backward[::-1]
    positions = n + np.arange(len(compact))
    plot_positions = positions + n+embedding-2
    available_stops = plot_positions+1
    scores = np.full(len(values), np.nan)
    sigmas = np.full((len(values), 2), np.nan)
    risks = np.full((len(values), 2, len(MULTIPLIERS)), np.nan)
    availability = np.full(len(values), -1, dtype=int)
    scores[positions] = compact
    sigmas[positions, 0], sigmas[positions, 1] = sigma_f, sigma_b[::-1]
    risks[positions, 0], risks[positions, 1] = risk_f, risk_b[::-1]
    availability[positions] = available_stops
    return scores, sigmas, risks, availability, plot_positions
