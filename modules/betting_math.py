# =============================================================================
# betting_math.py — Shared probability / edge / threshold logic
#
# Both odds.py (live picker) and historical_odds.py (backtest scorer) import
# from this module. Before this existed, the two files had quietly drifted:
# different min-edge thresholds, a confidence floor that only existed in one
# of them, a totals bias-correction that only existed in the other, and a
# crude linear run-line approximation in the backtest instead of the real
# Poisson calc used live. That meant the backtested win/loss record did NOT
# reflect what the live pipeline would actually have bet.
#
# Rule going forward: if you want to change an edge threshold, a confidence
# floor, or the totals bias correction, change it HERE ONCE. Never hardcode
# a threshold directly in odds.py or historical_odds.py again.
# =============================================================================

import numpy as np
from scipy.stats import poisson

# ── Shared thresholds ─────────────────────────────────────────────────────────
MIN_EDGE_ML       = 0.07   # minimum edge to flag a moneyline bet
MIN_EDGE_RL       = 0.07   # minimum edge to flag a run-line bet
MIN_EDGE_TOTAL    = 0.07   # minimum edge to flag a totals bet
ML_CONFIDENCE_MIN = 0.58   # don't fire an ML bet unless model win% is at least this
TOTALS_MIN_DISTANCE = 1.0  # skip totals bets where debiased xRuns is within 1 run of the line
TOTALS_BIAS_CORRECTION = 0.83  # model runs ~17% high vs actual — found via backtesting


# ── American odds <-> probability ─────────────────────────────────────────────

def american_to_prob(odds):
    """Convert American moneyline odds to implied probability."""
    if odds is None:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(p1: float, p2: float) -> tuple:
    """Remove bookmaker vig so two complementary probabilities sum to 1."""
    total = p1 + p2
    if total <= 0:
        return p1, p2
    return p1 / total, p2 / total


def payout_multiplier(odds):
    """Profit per $1 wagered (e.g. +150 -> 1.5, -150 -> 0.667).
    Clamped to a realistic range so bad data can't produce absurd payouts."""
    if odds is None:
        return None
    if odds > 0:
        return min(odds / 100, 10.0)
    return min(100 / abs(odds), 10.0)


# ── Poisson-based win/cover/total probabilities ───────────────────────────────

def poisson_over_prob(expected_total: float, line: float) -> float:
    """Probability that combined runs exceed the line."""
    lam_each = expected_total / 2
    threshold = int(np.floor(line))
    prob_over = 0.0
    for h in range(0, 20):
        for a in range(0, 20):
            if h + a > threshold:
                prob_over += poisson.pmf(h, lam_each) * poisson.pmf(a, lam_each)
    return prob_over


def poisson_cover_prob(lambda_home: float, lambda_away: float, spread: float) -> float:
    """Probability the 'home' side (as passed in) covers the given spread."""
    prob = 0.0
    margin_needed = abs(spread)
    for h in range(0, 20):
        for a in range(0, 20):
            if h - a > margin_needed:
                prob += (poisson.pmf(h, max(0.1, lambda_home)) *
                         poisson.pmf(a, max(0.1, lambda_away)))
    return prob


def debiased_total(model_xruns: float) -> float:
    """Apply the totals bias correction discovered via backtesting
    (raw model xRuns runs ~17% high on average vs actual)."""
    return model_xruns * TOTALS_BIAS_CORRECTION
