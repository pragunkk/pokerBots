#!/usr/bin/env python3
"""
SparBot v2 — Complementary ML + Quantile-Conditioned Bayesian EV Bot for Discard Hold'em.
Built in `spar/spar_bot.py` to spar against our flagship `Starter_Kit/bot.py`.

Key Upgrades in SparBot v2 over v1:
  1. Pre-Trained 2-Layer Neural/Logistic Response Network (`NeuralResponseModel`) + Online SGD:
     - Predicts `P(fold | x)` and `P(raise | x, not fold)` across a 12-feature postflop state vector
       and updates output weights live during the match via Stochastic Gradient Descent (SGD).
  2. Bayesian Quantile-Conditioned Adverse Selection in Multi-Sizing Action EV Tree:
     - Replaces v1's linear equity shrinkage with exact quantile truncation:
       `E_called(B) = (E_raw - quantile_cut) / (1 - quantile_cut)`
       This prevents overbetting (`1.15x` pot) across 3 streets with one-pair hands while naturally
       selecting geometric/overbet sizing (`0.80x-1.10x` pot) with nutted hands (`E_raw >= 0.84`).
  3. Postflop Raise-Polarization & Paired-Board Exponent Boost (`_agg_k` v2):
     - Sharpens the Bayesian range exponent `k` when facing a postflop raise over our bet (`my_r >= 1`)
       or multi-street barrels on paired/flush boards, eliminating baby-flush 3-bet punts and
       paired-board underpair call-downs.
"""

import json
import math
import os as _os
import random
import sys
import time
from bisect import bisect_left, bisect_right
from itertools import combinations, combinations_with_replacement

# ---------------------------------------------------------------------------
# CARDS & FAST LOOKUP-TABLE HAND EVALUATOR (Built in 5s free startup window)
# ---------------------------------------------------------------------------
RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_TO_INT = {r: i for i, r in enumerate(RANKS)}
SUIT_TO_INT = {s: i for i, s in enumerate(SUITS)}
INT_TO_CARD = [r + s for r in RANKS for s in SUITS]
CARD_TO_INT = {c: i for i, c in enumerate(INT_TO_CARD)}
CARD_RANK = [i // 4 for i in range(52)]
CARD_SUIT = [i % 4 for i in range(52)]
CARD_BIT = [1 << (i // 4) for i in range(52)]

BIG_BLIND = 2

POW5 = [5 ** i for i in range(13)]
CARD_POW5 = [POW5[i // 4] for i in range(52)]
SUITADD = [1 << (4 * (i & 3)) for i in range(52)]

NON_FLUSH_LUT = {}
FLUSH_BIT_LUT = {}


def _score_ranks(ranks_tuple, is_flush):
    counts = {}
    for r in ranks_tuple:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [g[1] for g in groups]
    ordered = [g[0] for g in groups]
    straight_high = -1
    if len(groups) == 5:
        hi, lo = ordered[0], ordered[4]
        if hi - lo == 4:
            straight_high = hi
        elif ordered == [12, 3, 2, 1, 0]:
            straight_high = 3
    if is_flush and straight_high >= 0:
        cat, tie = 8, (straight_high, 0, 0, 0, 0)
    elif shape == [4, 1]:
        cat, tie = 7, (ordered[0], ordered[1], 0, 0, 0)
    elif shape == [3, 2]:
        cat, tie = 6, (ordered[0], ordered[1], 0, 0, 0)
    elif is_flush:
        cat, tie = 5, tuple(ordered)
    elif straight_high >= 0:
        cat, tie = 4, (straight_high, 0, 0, 0, 0)
    elif shape == [3, 1, 1]:
        cat, tie = 3, (ordered[0], ordered[1], ordered[2], 0, 0)
    elif shape == [2, 2, 1]:
        cat, tie = 2, (ordered[0], ordered[1], ordered[2], 0, 0)
    elif shape == [2, 1, 1, 1]:
        cat, tie = 1, (ordered[0], ordered[1], ordered[2], ordered[3], 0)
    else:
        cat, tie = 0, tuple(ordered)
    return (cat << 20) | (tie[0] << 16) | (tie[1] << 12) | (tie[2] << 8) | (tie[3] << 4) | tie[4]


def _init_tables():
    for r0 in range(13):
        p0 = POW5[r0]
        for r1 in range(r0, 13):
            p1 = p0 + POW5[r1]
            for r2 in range(r1, 13):
                p2 = p1 + POW5[r2]
                for r3 in range(r2, 13):
                    p3 = p2 + POW5[r3]
                    for r4 in range(r3, 13):
                        if r0 == r4:
                            continue
                        key = p3 + POW5[r4]
                        rt = (r0, r1, r2, r3, r4)
                        NON_FLUSH_LUT[key] = _score_ranks(rt, False)
                        if r0 < r1 < r2 < r3 < r4:
                            mask = (1 << r0) | (1 << r1) | (1 << r2) | (1 << r3) | (1 << r4)
                            FLUSH_BIT_LUT[mask] = _score_ranks(rt, True)


_init_tables()
COMBOS_7_5 = tuple(combinations(range(7), 5))
COMBOS_6_5 = tuple(combinations(range(6), 5))

N7 = {}
N6 = {}
FL = {}
FSUIT = [-1] * 65536


def _init_multi():
    nf = NON_FLUSH_LUT
    for combo in combinations_with_replacement(range(13), 7):
        if combo[0] == combo[4] or combo[1] == combo[5] or combo[2] == combo[6]:
            continue
        p = [POW5[r] for r in combo]
        best = 0
        for i0, i1, i2, i3, i4 in COMBOS_7_5:
            v = nf[p[i0] + p[i1] + p[i2] + p[i3] + p[i4]]
            if v > best:
                best = v
        N7[sum(p)] = best
    for combo in combinations_with_replacement(range(13), 6):
        if combo[0] == combo[4] or combo[1] == combo[5]:
            continue
        p = [POW5[r] for r in combo]
        best = 0
        for i0, i1, i2, i3, i4 in COMBOS_6_5:
            v = nf[p[i0] + p[i1] + p[i2] + p[i3] + p[i4]]
            if v > best:
                best = v
        N6[sum(p)] = best
    fb = FLUSH_BIT_LUT
    for mask in range(8192):
        bits = [i for i in range(13) if (mask >> i) & 1]
        if len(bits) < 5:
            continue
        best = 0
        for sub in combinations(bits, 5):
            v = fb[(1 << sub[0]) | (1 << sub[1]) | (1 << sub[2]) | (1 << sub[3]) | (1 << sub[4])]
            if v > best:
                best = v
        FL[mask] = best
    for s in range(65536):
        for su in range(4):
            if ((s >> (4 * su)) & 15) >= 5:
                FSUIT[s] = su
                break


_init_multi()


def eval_5(c0, c1, c2, c3, c4):
    if CARD_SUIT[c0] == CARD_SUIT[c1] == CARD_SUIT[c2] == CARD_SUIT[c3] == CARD_SUIT[c4]:
        return FLUSH_BIT_LUT[CARD_BIT[c0] | CARD_BIT[c1] | CARD_BIT[c2] | CARD_BIT[c3] | CARD_BIT[c4]]
    return NON_FLUSH_LUT[CARD_POW5[c0] + CARD_POW5[c1] + CARD_POW5[c2] + CARD_POW5[c3] + CARD_POW5[c4]]


def eval_7(cs):
    a, b, c, d, e, f, g = cs
    v = N7[CARD_POW5[a] + CARD_POW5[b] + CARD_POW5[c] + CARD_POW5[d] + CARD_POW5[e] + CARD_POW5[f] + CARD_POW5[g]]
    fs = FSUIT[SUITADD[a] + SUITADD[b] + SUITADD[c] + SUITADD[d] + SUITADD[e] + SUITADD[f] + SUITADD[g]]
    if fs >= 0:
        m = 0
        for x in cs:
            if (x & 3) == fs:
                m |= CARD_BIT[x]
        fv = FL[m]
        if fv > v:
            v = fv
    return v


def eval_6(cs):
    a, b, c, d, e, f = cs
    v = N6[CARD_POW5[a] + CARD_POW5[b] + CARD_POW5[c] + CARD_POW5[d] + CARD_POW5[e] + CARD_POW5[f]]
    fs = FSUIT[SUITADD[a] + SUITADD[b] + SUITADD[c] + SUITADD[d] + SUITADD[e] + SUITADD[f]]
    if fs >= 0:
        m = 0
        for x in cs:
            if (x & 3) == fs:
                m |= CARD_BIT[x]
        fv = FL[m]
        if fv > v:
            v = fv
    return v


def eval_board_hand(h0, h1, board):
    n = len(board)
    if n == 5:
        return eval_7((h0, h1) + tuple(board))
    if n == 4:
        return eval_6((h0, h1) + tuple(board))
    return eval_5(h0, h1, board[0], board[1], board[2])


def flop_key(a, b, flop):
    sc = eval_5(a, b, flop[0], flop[1], flop[2])
    sa, sb = CARD_SUIT[a], CARD_SUIT[b]
    for s in (sa, sb) if sa != sb else (sa,):
        hs = (sa == s) + (sb == s)
        bs = (CARD_SUIT[flop[0]] == s) + (CARD_SUIT[flop[1]] == s) + (CARD_SUIT[flop[2]] == s)
        if hs + bs == 4:
            sc += 1 << 18
    return sc


def greedy_keep(o0, o1, o2, flop):
    k01 = flop_key(o0, o1, flop)
    k02 = flop_key(o0, o2, flop)
    k12 = flop_key(o1, o2, flop)
    if k12 >= k01 and k12 >= k02:
        return o1, o2, k12
    if k02 >= k01:
        return o0, o2, k02
    return o0, o1, k01


def board_danger_int(board_ints):
    if len(board_ints) < 3:
        return 0.0
    suits = [CARD_SUIT[c] for c in board_ints]
    ranks = sorted({CARD_RANK[c] + 2 for c in board_ints})
    if 14 in ranks:
        ranks = [1] + ranks
    d = 0.0
    ms = max(suits.count(s) for s in range(4))
    if ms >= 4:
        d += 0.55
    elif ms == 3:
        d += 0.25
    mc = 1
    for i in range(len(ranks)):
        c = sum(1 for r in ranks if 0 <= r - ranks[i] <= 4)
        if c > mc:
            mc = c
    if mc >= 4:
        d += 0.40
    elif mc == 3 and len(board_ints) >= 4:
        d += 0.15
    if len({CARD_RANK[c] for c in board_ints}) < len(board_ints):
        d += 0.15
    return min(1.0, d)


# ---------------------------------------------------------------------------
# PREFLOP TABLE & 20x10 HAND-VS-RANGE EQUITY MATRIX
# ---------------------------------------------------------------------------
sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "Starter_Kit"))
from preflop_table import PREFLOP_TABLE_1755


def canonical_key_ints(hole_ints):
    idxs = sorted(hole_ints, key=lambda i: CARD_RANK[i], reverse=True)
    r0, r1, r2 = CARD_RANK[idxs[0]] + 2, CARD_RANK[idxs[1]] + 2, CARD_RANK[idxs[2]] + 2
    s0, s1, s2 = CARD_SUIT[idxs[0]], CARD_SUIT[idxs[1]], CARD_SUIT[idxs[2]]
    if r0 == r1 == r2:
        sp = 0
    elif r0 == r1:
        sp = 1 if (s2 == s0 or s2 == s1) else 0
    elif r1 == r2:
        sp = 1 if (s0 == s1 or s0 == s2) else 0
    else:
        if s0 == s1 == s2:
            sp = 4
        elif s0 == s1:
            sp = 3
        elif s0 == s2:
            sp = 2
        elif s1 == s2:
            sp = 1
        else:
            sp = 0
    return (r0, r1, r2, sp)


_SORTED_PF_VALS = sorted(PREFLOP_TABLE_1755.values())
_N_PF = len(_SORTED_PF_VALS)
PF_PCT_MAP = {
    k: (bisect_left(_SORTED_PF_VALS, v) + bisect_right(_SORTED_PF_VALS, v)) * 0.5 / _N_PF
    for k, v in PREFLOP_TABLE_1755.items()
}


def preflop_pct(hole_ints):
    return PF_PCT_MAP.get(canonical_key_ints(hole_ints), 0.50)


RANGE_FS = [0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.45, 0.62, 0.80, 1.00]
RANGE_RAW = (
    "202,209,230,268,290,314,318,331,340,361;216,216,242,277,306,323,341,348,371,383;"
    "215,215,248,283,309,332,352,355,379,411;206,217,250,286,320,339,359,370,387,418;"
    "209,216,258,297,320,341,358,375,392,429;209,222,263,296,321,346,366,388,405,444;"
    "210,227,272,303,340,357,372,397,422,451;211,230,275,306,331,353,371,404,431,464;"
    "215,243,285,319,346,361,386,416,441,472;214,255,292,325,346,369,391,426,453,480;"
    "216,251,290,321,349,376,402,436,466,496;214,250,301,333,358,381,410,441,476,506;"
    "215,269,317,341,363,392,425,465,490,518;223,270,313,344,370,402,437,475,499,529;"
    "223,271,319,350,379,422,452,482,518,538;228,300,343,362,400,440,476,507,534,556;"
    "240,294,343,382,423,467,506,533,549,570;255,324,366,406,455,498,532,556,566,587;"
    "222,272,370,439,489,524,563,586,610,638;355,550,628,653,674,699,715,723,741,747"
)
RANGE_M = [[int(x) / 1000.0 for x in row.split(",")] for row in RANGE_RAW.split(";")]


def range_equity(p, f):
    nb = len(RANGE_M)
    x = min(nb - 1.0, max(0.0, p * nb - 0.5))
    b0 = int(x)
    b1 = min(nb - 1, b0 + 1)
    tx = x - b0
    if f <= RANGE_FS[0]:
        j0 = j1 = 0
        tf = 0.0
    elif f >= RANGE_FS[-1]:
        j0 = j1 = len(RANGE_FS) - 1
        tf = 0.0
    else:
        j1 = bisect_left(RANGE_FS, f)
        j0 = j1 - 1
        tf = (f - RANGE_FS[j0]) / (RANGE_FS[j1] - RANGE_FS[j0])
    r0 = RANGE_M[b0][j0] * (1 - tf) + RANGE_M[b0][j1] * tf
    r1 = RANGE_M[b1][j0] * (1 - tf) + RANGE_M[b1][j1] * tf
    return r0 * (1 - tx) + r1 * tx


# ---------------------------------------------------------------------------
# PRE-TRAINED 2-LAYER NEURAL RESPONSE MODEL + ONLINE SGD ADAPTATION
# ---------------------------------------------------------------------------
def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 20.0)))
    ez = math.exp(max(z, -20.0))
    return ez / (1.0 + ez)


class NeuralResponseModel:
    def __init__(self):
        self.w_fold = [-1.11587, 0.10502, 0.03239, 0.03066, -0.03704, 2.34967, -0.26091, -0.15572, -0.2894, 0.37678, -0.03297, -0.23504]
        self.w_raise = [-0.83914, -0.41882, -0.92343, -0.30257, 0.23153, -0.68964, 1.26116, 0.03423, -0.05194, -0.10207, -0.13377, -0.56624]
        self.W_h = [
            [0.84341, -0.01269, -0.17935, -0.07798, 0.1222, -1.10145, -0.22486, -0.14608, 0.09779, -0.20145, -0.01924, -0.15935],
            [-0.17965, -0.14236, 0.01768, 0.25886, 0.17217, 0.62921, -0.47768, 0.03591, -0.06356, -0.1731, 0.01151, 0.05959],
            [0.66789, -0.06289, 0.11761, -0.57835, -0.17594, -0.29284, -0.36979, 0.04286, -0.05799, 0.0703, 0.1797, -0.05067],
            [-0.14396, -0.01249, -0.05825, -0.13183, -0.07279, 0.10885, -0.00375, -0.04942, 0.12986, 0.12046, 0.01256, -0.23021],
            [-0.76273, 0.0442, -0.01477, 0.19292, 0.20754, 0.49888, 0.04128, 0.01771, -0.01213, 0.16658, 0.19104, -0.02064],
            [0.3532, -0.16808, -0.12853, -0.21707, 0.07969, -0.96214, 0.61032, -0.02831, -0.02782, -0.10084, 0.30642, -0.12274],
        ]
        self.u_fold = [-0.98034, 0.42555, -0.53194, 0.11886, 0.61807, -0.68926]
        self.u_raise = [-0.26514, -0.5351, -0.67875, 0.1806, 0.23338, 0.83998]
        self.updates = 0

    def build_features(self, street_idx, bet_size, pot, danger, opp_fold_rate, opp_agg_rate, opp_vpip, is_3bet, opp_checked, opp_hi_disc):
        bet_to_pot = min(2.5, max(0.05, bet_size / max(1.0, float(pot))))
        pot_odds = bet_size / max(1.0, float(pot + 2.0 * bet_size))
        pot_commit = min(1.0, (pot + bet_size) / 200.0)
        return [
            1.0,
            street_idx / 3.0,
            bet_to_pot,
            pot_odds,
            danger,
            opp_fold_rate,
            opp_agg_rate,
            opp_vpip,
            1.0 if is_3bet else 0.0,
            1.0 if opp_checked else 0.0,
            1.0 if opp_hi_disc else 0.0,
            pot_commit,
        ]

    def predict(self, x):
        h = [math.tanh(max(-15.0, min(15.0, sum(row[i] * x[i] for i in range(12))))) for row in self.W_h]
        zf = sum(self.w_fold[i] * x[i] for i in range(12)) + sum(self.u_fold[j] * h[j] for j in range(6))
        zr = sum(self.w_raise[i] * x[i] for i in range(12)) + sum(self.u_raise[j] * h[j] for j in range(6))
        p_fold = _sigmoid(zf)
        p_raise_cond = _sigmoid(zr)
        p_raise = (1.0 - p_fold) * p_raise_cond
        p_call = max(0.0, 1.0 - p_fold - p_raise)
        return p_fold, p_call, p_raise, p_raise_cond, h

    def online_sgd_update(self, x, opp_action):
        p_fold, _, _, p_raise_cond, h = self.predict(x)
        y_fold = 1.0 if opp_action == "fold" else 0.0
        self.updates += 1
        lr = 0.075 / (1.0 + 0.015 * self.updates)

        err_f = p_fold - y_fold
        for i in range(12):
            self.w_fold[i] -= lr * err_f * x[i]
        for j in range(6):
            self.u_fold[j] -= lr * err_f * h[j]

        if opp_action in ("call", "raise"):
            y_raise = 1.0 if opp_action == "raise" else 0.0
            err_r = p_raise_cond - y_raise
            for i in range(12):
                self.w_raise[i] -= lr * err_r * x[i]
            for j in range(6):
                self.u_raise[j] -= lr * err_r * h[j]


# ---------------------------------------------------------------------------
# BAYESIAN OPPONENT TRACKER & EQUITY ENGINE
# ---------------------------------------------------------------------------
class BayesianOpponentTracker:
    def __init__(self):
        self.hands = 0
        self.pre_n = 0
        self.pre_vpip = 0
        self.pre_raise = 0
        self.open_n = 0
        self.open_r = 0
        self.f3_n = 0
        self.f3_r = 0
        self.f3_fold = 0
        self.f4_n = 0
        self.f4_r = 0
        self.f4_fold = 0
        self.post_actions = 0
        self.post_raises = 0
        self.faced_bet = 0
        self.folded_to_bet = 0
        self.called_bet = 0
        self.raised_bet = 0
        self.k_sum = 0.0
        self.k_n = 0
        self.disc_n = 0
        self.disc_greedy = 0

    @property
    def vpip(self):
        return (self.pre_vpip + 5.5) / (self.pre_n + 10.0)

    @property
    def f_open(self):
        return (self.open_r + 2.7) / (self.open_n + 6.0)

    @property
    def f_3bet(self):
        return (self.f3_r + 0.8) / (self.f3_n + 8.0)

    @property
    def f_4bet(self):
        return (self.f4_r + 0.48) / (self.f4_n + 8.0)

    @property
    def fold_to_open(self):
        return (self.f3_fold + 2.4) / (self.f3_n + 6.0)

    @property
    def fold_to_3bet(self):
        return (self.f4_fold + 2.4) / (self.f4_n + 6.0)

    @property
    def fold_to_bet(self):
        return (self.folded_to_bet + 3.2) / (self.faced_bet + 8.0)

    @property
    def raise_vs_bet(self):
        return (self.raised_bet + 0.8) / (self.faced_bet + 8.0)

    @property
    def agg(self):
        return (self.post_raises + 2.0) / (self.post_actions + 8.0)

    @property
    def mean_bet_pct(self):
        return (self.k_sum + 0.72 * 4.0) / (self.k_n + 4.0)

    @property
    def k_base(self):
        m = min(0.93, max(0.50, self.mean_bet_pct))
        return max(0.0, min(4.5, (2.0 * m - 1.0) / (1.0 - m)))

    @property
    def inconsistent_weight(self):
        obs = (self.disc_greedy + 4.8) / (self.disc_n + 6.0)
        p = max(0.0, min(0.97, (obs - 0.33) / 0.67))
        q = (1.0 - p) / 3.0
        return max(0.04, min(1.0, q / (p + q)))


PRE_FLOOR = 0.10


def _pct_ranks(scores):
    srt = sorted(scores)
    n = len(srt)
    return [(bisect_left(srt, s) + bisect_right(srt, s)) * 0.5 / n for s in scores]


def post_equity_ml(hole2, board, dead_extra, opp_disc, k, kc, inc_w, deadline, pre_f=None, min_cnt=140):
    dead = set(hole2) | set(board) | set(dead_extra)
    rem = [c for c in range(52) if c not in dead]
    pairs = list(combinations(rem, 2))
    flop = board[:3]
    bt = tuple(board)
    h = (hole2[0], hole2[1])
    n = len(bt)
    if n == 5:
        my = eval_7(h + bt)
        sc = [eval_7(p + bt) for p in pairs]
    else:
        my = None
        sc = [eval_6(p + bt) for p in pairs]
    pcts = _pct_ranks(sc)
    weights = []
    for i, (a, b) in enumerate(pairs):
        w = 1.0
        if k > 0:
            w *= (pcts[i] + 0.03) ** k
        if kc > 0:
            w *= (1.05 - pcts[i]) ** kc
        if opp_disc is not None:
            kept = flop_key(a, b, flop)
            alt = max(flop_key(a, opp_disc, flop), flop_key(b, opp_disc, flop))
            if kept < alt:
                w *= inc_w
            if pre_f is not None:
                if preflop_pct((a, b, opp_disc)) < 1.0 - pre_f:
                    w *= PRE_FLOOR
        weights.append(w)
    if n == 5:
        tw = 0.0
        ww = 0.0
        for i, w in enumerate(weights):
            tw += w
            s = sc[i]
            if my > s:
                ww += w
            elif my == s:
                ww += 0.5 * w
        return ww / max(tw, 1e-12)

    cum = []
    t = 0.0
    for w in weights:
        t += w
        cum.append(t)
    if t <= 0:
        return 0.5
    rnd = random.random
    wins = 0.0
    cnt = 0
    npairs = len(pairs)
    nrem = len(rem)
    clock = time.perf_counter
    while True:
        for _ in range(32):
            idx = bisect_left(cum, rnd() * t)
            if idx >= npairs:
                idx = npairs - 1
            a, b = pairs[idx]
            while True:
                r = rem[int(rnd() * nrem)]
                if r != a and r != b:
                    break
            full = bt + (r,)
            ms = eval_7(h + full)
            os_ = eval_7((a, b) + full)
            if ms > os_:
                wins += 1.0
            elif ms == os_:
                wins += 0.5
            cnt += 1
        if (clock() >= deadline and cnt >= min_cnt) or cnt >= 4500:
            break
    return wins / cnt


def flop_equities_ml(hole3, flop, dead_extra, k, deadline, min_samples=100, max_samples=2500, pre_f=None):
    dead = set(hole3) | set(flop) | set(dead_extra)
    rem = [c for c in range(52) if c not in dead]
    f = tuple(flop)
    c0, c1, c2 = hole3
    opts = ((c1, c2), (c0, c2), (c0, c1))
    sample = random.sample
    rnd = random.random
    clock = time.perf_counter
    okeys = []
    res = [[], [], []]
    n = 0
    while True:
        for _ in range(24):
            o0, o1, o2, t, r = sample(rem, 5)
            if pre_f is not None:
                tries = 0
                while tries < 8 and preflop_pct((o0, o1, o2)) < 1.0 - pre_f and rnd() > PRE_FLOOR:
                    o0, o1, o2, t, r = sample(rem, 5)
                    tries += 1
            a, b, key = greedy_keep(o0, o1, o2, f)
            full = f + (t, r)
            os_ = eval_7((a, b) + full)
            okeys.append(key)
            for i in range(3):
                ms = eval_7(opts[i] + full)
                res[i].append(1.0 if ms > os_ else (0.5 if ms == os_ else 0.0))
            n += 1
        if (clock() >= deadline and n >= min_samples) or n >= max_samples:
            break
    if k > 0:
        pcts = _pct_ranks(okeys)
        ws = [(p + 0.03) ** k for p in pcts]
    else:
        ws = [1.0] * n
    tw = sum(ws)
    return [sum(ws[j] * res[i][j] for j in range(n)) / tw for i in range(3)]


# ---------------------------------------------------------------------------
# ML-EV SPARRING BOT CONTROLLER (v2)
# ---------------------------------------------------------------------------
class MLSparBot:
    def __init__(self):
        self.total_hands = 200
        self.bank_ms = 30000.0
        self.time_left_ms = 30000.0
        self.opp = BayesianOpponentTracker()
        self.nn = NeuralResponseModel()

        self.hand_id = 0
        self.seat = "button"
        self.hole_cards = []
        self.my_discard = None
        self.opp_discard = None
        self.reached = 0
        self.seen = 0
        self.flop_opp_raises = 0
        self.flop_bet_frac = 0.0
        self.pre_f = None

        self.pending_bet_features = None
        self.processed_hist_len = 0
        self.cal_n = 0
        self.cal_gap = 0.0
        self.pend_river_est = None

    def handle_init(self, msg):
        self.total_hands = int(msg.get("total_hands", 200))
        self.time_left_ms = float(msg.get("time_bank_ms", 30000))
        self.bank_ms = self.time_left_ms

    def handle_new_hand(self, msg):
        self.hand_id = msg.get("hand_id", 0)
        self.seat = msg.get("seat", "button")
        self.hole_cards = list(msg.get("hole_cards", []))
        self.my_discard = None
        self.opp_discard = None
        self.flop_opp_raises = 0
        self.flop_bet_frac = 0.0
        self.seen += 1
        self.pre_f = None
        self.pending_bet_features = None
        self.processed_hist_len = 0
        self.pend_river_est = None

    def handle_discard_result(self, msg):
        self.my_discard = msg.get("my_discard")
        self.opp_discard = msg.get("opp_discard")
        self.hole_cards = list(msg.get("hole_cards", []))

    def _ingest_new_history_for_sgd(self, history):
        new_acts = history[self.processed_hist_len :]
        self.processed_hist_len = len(history)
        for h in new_acts:
            if h.get("actor") == "opp" and h.get("street") != "preflop":
                act = h.get("action")
                if self.pending_bet_features is not None and act in ("fold", "call", "raise"):
                    self.nn.online_sgd_update(self.pending_bet_features, act)
                    self.pending_bet_features = None

    def handle_hand_result(self, msg):
        hist = msg.get("history", [])
        self._ingest_new_history_for_sgd(hist)
        net = msg.get("net", 0)
        if self.pend_river_est is not None and msg.get("result") == "showdown":
            win = 1.0 if net > 0 else (0.5 if net == 0 else 0.0)
            self.cal_n += 1
            self.cal_gap += win - self.pend_river_est
        self.pend_river_est = None

        self.opp.hands += 1
        i_am_btn = (self.seat == "button")
        r = 0
        for h in hist:
            if h.get("street") != "preflop":
                break
            actor, act = h.get("actor"), h.get("action")
            if actor == "opp":
                if r == 0 and not i_am_btn:
                    self.opp.open_n += 1
                    if act == "raise":
                        self.opp.open_r += 1
                elif r == 1:
                    self.opp.f3_n += 1
                    if act == "raise":
                        self.opp.f3_r += 1
                    elif act == "fold":
                        self.opp.f3_fold += 1
                elif r == 2:
                    self.opp.f4_n += 1
                    if act == "raise":
                        self.opp.f4_r += 1
                    elif act == "fold":
                        self.opp.f4_fold += 1
            if act == "raise":
                r += 1

        opp_saw = opp_vpip = opp_pfr = opp_raised_post = False
        my_bet_st = None
        for h in hist:
            st, actor, act = h.get("street"), h.get("actor"), h.get("action")
            if st == "preflop":
                if actor == "opp":
                    opp_saw = True
                    if act in ("call", "raise"):
                        opp_vpip = True
                    if act == "raise":
                        opp_pfr = True
            else:
                if actor == "me" and act == "raise":
                    my_bet_st = st
                elif actor == "opp":
                    self.opp.post_actions += 1
                    if act == "raise":
                        self.opp.post_raises += 1
                        opp_raised_post = True
                    if my_bet_st == st:
                        self.opp.faced_bet += 1
                        if act == "fold":
                            self.opp.folded_to_bet += 1
                        elif act == "call":
                            self.opp.called_bet += 1
                        elif act == "raise":
                            self.opp.raised_bet += 1
                        my_bet_st = None
        if opp_saw:
            self.opp.pre_n += 1
            if opp_vpip:
                self.opp.pre_vpip += 1
            if opp_pfr:
                self.opp.pre_raise += 1

        oh = msg.get("opp_hole_cards")
        board = msg.get("board") or []
        mine = msg.get("my_hole_cards") or []
        if oh and len(oh) == 2 and len(board) == 5 and len(mine) == 2:
            try:
                o = [CARD_TO_INT[c] for c in oh]
                b = [CARD_TO_INT[c] for c in board]
                m = [CARD_TO_INT[c] for c in mine]
                extra = [CARD_TO_INT[d] for d in (self.my_discard, self.opp_discard) if d in CARD_TO_INT]
                if opp_raised_post:
                    rem = [c for c in range(52) if c not in (set(b) | set(m) | set(extra))]
                    osc = eval_7((o[0], o[1]) + tuple(b))
                    lo = hi = 0
                    bt = tuple(b)
                    for x, y in combinations(rem, 2):
                        s = eval_7((x, y) + bt)
                        if s < osc:
                            lo += 1
                        elif s == osc:
                            hi += 1
                    tot = len(rem) * (len(rem) - 1) // 2
                    if tot:
                        self.opp.k_sum += (lo + 0.5 * hi) / tot
                        self.opp.k_n += 1
                if self.opp_discard in CARD_TO_INT and len(b) >= 3:
                    d = CARD_TO_INT[self.opp_discard]
                    flop = b[:3]
                    kept = flop_key(o[0], o[1], flop)
                    alt = max(flop_key(o[0], d, flop), flop_key(o[1], d, flop))
                    self.opp.disc_n += 1
                    if kept >= alt:
                        self.opp.disc_greedy += 1
            except Exception:
                pass

    def _deadline(self, weight, cap_ms=500.0):
        left = self.time_left_ms
        hands_left = max(1, self.total_hands - self.hand_id + 1)
        reach = max(0.35, (self.reached + 6.0) / (self.seen + 10.0))
        reserve = max(3000.0, 0.10 * self.bank_ms)
        per_hand = max(0.0, 0.50 * (left - reserve) / hands_left)
        budget = min(cap_ms, per_hand / reach * weight) if left >= reserve else 0.0
        return time.perf_counter() + budget / 1000.0

    def _floor(self):
        if self.time_left_ms < max(3000.0, 0.10 * self.bank_ms):
            return 16
        if self.time_left_ms < 0.30 * self.bank_ms:
            return 32
        return 48

    def _river_cal_bias(self):
        if self.cal_n < 4:
            return 0.0
        return max(-0.18, min(0.08, self.cal_gap / (self.cal_n + 6.0)))

    @staticmethod
    def _safe(action, amount, req):
        legal = req.get("legal_actions", [])
        if action not in legal:
            if "check" in legal:
                return {"type": "action", "action": "check"}
            if "call" in legal and req.get("to_call", 0) <= 2:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold" if "fold" in legal else legal[0]}
        if action == "raise":
            lo = int(req.get("min_raise", 2))
            hi = int(req.get("max_raise", 200))
            return {"type": "action", "action": "raise", "amount": max(lo, min(hi, int(round(amount))))}
        return {"type": "action", "action": action}

    def _agg_k(self, history, street, opp_bet, my_bet, pot):
        kb = self.opp.k_base
        opp_r = sum(1 for h in history if h.get("street") == street and h.get("actor") == "opp" and h.get("action") == "raise")
        my_r = sum(1 for h in history if h.get("street") == street and h.get("actor") == "me" and h.get("action") == "raise")
        prior_r = sum(1 for h in history if h.get("street") not in (street, "preflop") and h.get("actor") == "opp" and h.get("action") == "raise")
        if opp_r > 0 and opp_bet > 0:
            pot_before = max(2.0, pot - opp_bet - my_bet)
            frac = opp_bet / pot_before
            sf = 0.55 + 0.72 * min(frac, 1.6)
            k = kb * sf * (1.0 + 0.35 * (opp_r - 1)) * (1.0 + 0.25 * prior_r)
            # v2 upgrade: if opponent raised over our bet on this street (`my_r >= 1`), their range is heavily polarized to strong hands
            if my_r >= 1 and opp_bet > my_bet:
                k += 1.15
        else:
            k = kb * 0.30 * prior_r
        return min(k, 6.5)

    def _opp_pre_range(self, history):
        pre = [h for h in history if h.get("street") == "preflop"]
        raises = [h for h in pre if h.get("action") == "raise"]
        if not raises:
            return None
        O = self.opp
        level = len(raises)
        if raises[-1].get("actor") == "opp":
            if level == 1:
                f = O.f_open if self.seat != "button" else 0.40
            elif level == 2:
                f = O.f_3bet
            elif level == 3:
                f = O.f_4bet
            else:
                f = 0.04
            return max(0.04, min(1.0, f))
        if level == 1:
            f = 1.0 - O.fold_to_open
        elif level == 2:
            f = 1.0 - O.fold_to_3bet
        else:
            f = 0.30
        return max(0.15, min(1.0, f))

    def choose_discard(self, req):
        hole = req.get("hole_cards", self.hole_cards)
        board = req.get("board", [])
        self.reached += 1
        if len(hole) != 3 or len(board) < 3:
            worst = min(hole, key=lambda c: (RANK_TO_INT[c[0]], SUIT_TO_INT[c[1]]))
            return {"type": "discard", "card": worst}
        h = [CARD_TO_INT[c] for c in hole]
        f = [CARD_TO_INT[c] for c in board[:3]]
        k = 0.0
        if self.flop_opp_raises > 0:
            k = self.opp.k_base * (0.55 + 0.70 * min(self.flop_bet_frac, 1.6))
        dl = self._deadline(0.38, cap_ms=650.0)
        eqs = flop_equities_ml(h, f, [], k, dl, min_samples=self._floor(), max_samples=4000, pre_f=self.pre_f)
        best = max(range(3), key=lambda i: (round(eqs[i], 5), -CARD_RANK[h[i]]))
        return {"type": "discard", "card": hole[best]}

    def choose_action(self, req):
        if "hole_cards" in req and req["hole_cards"]:
            self.hole_cards = list(req["hole_cards"])
        if "seat" in req:
            self.seat = req["seat"]
        hist = req.get("history", [])
        self._ingest_new_history_for_sgd(hist)
        street = req.get("street", "preflop")
        if street == "preflop":
            return self._play_preflop_ml(req)
        return self._play_postflop_ml(req)

    def _play_preflop_ml(self, req):
        legal = req.get("legal_actions", [])
        to_call = req.get("to_call", 0)
        pot = req.get("pot", 3)
        opp_bet = req.get("opp_bet", 0)
        history = req.get("history", [])
        hole = [CARD_TO_INT[c] for c in self.hole_cards]
        p = preflop_pct(hole)
        opp_r = sum(1 for h in history if h.get("street") == "preflop" and h.get("actor") == "opp" and h.get("action") == "raise")
        my_r = sum(1 for h in history if h.get("street") == "preflop" and h.get("actor") == "me" and h.get("action") == "raise")
        O = self.opp
        i_am_button = (self.seat == "button")

        if opp_bet <= BIG_BLIND and opp_r == 0:
            if i_am_button or to_call > 0:
                open_p = 0.36
                if O.hands >= 10 and O.fold_to_open > 0.54:
                    open_p = 0.12
                elif O.hands >= 10 and O.f_3bet > 0.24:
                    open_p = 0.48
                if p >= open_p and "raise" in legal:
                    size = 6 if p >= 0.75 else 5
                    return self._safe("raise", size, req)
                if p >= 0.12 and "call" in legal:
                    return self._safe("call", None, req)
                return self._safe("fold", None, req)
            iso_p = 0.52
            if O.hands >= 10 and O.fold_to_open > 0.50:
                iso_p = 0.25
            if p >= iso_p and "raise" in legal:
                size = 8 if p >= 0.85 else 6
                return self._safe("raise", size, req)
            return self._safe("check", None, req)

        level = my_r + opp_r
        if level <= 1:
            f = O.f_open
        elif level == 2:
            f = O.f_3bet
        elif level == 3:
            f = O.f_4bet
        else:
            f = 0.03
        f = max(0.03, min(1.0, f))
        if opp_bet >= 40:
            f = min(f, 0.12)
        elif opp_bet >= 24:
            f = min(f, 0.22)

        E = range_equity(p, f)
        pot_odds = to_call / max(1.0, float(pot + to_call))
        margin = 0.030 + (0.025 if not i_am_button else 0.0) + 0.012 * max(0, level - 1)
        if to_call >= req.get("my_stack", 200) - 0.5:
            margin = 0.005

        if "raise" in legal:
            thr3 = 0.54 if (O.hands >= 8 and O.f_open >= 0.58) else 0.61
            if level == 1 and my_r == 0 and E >= thr3 and opp_bet <= 28:
                return self._safe("raise", min(55, opp_bet * 3), req)
            if level == 1 and my_r == 0 and O.hands >= 8 and O.fold_to_3bet > 0.60 and p >= 0.45 and opp_bet <= 10:
                if random.random() < 0.45:
                    return self._safe("raise", min(55, opp_bet * 3), req)
            if level == 2 and E >= 0.64:
                return self._safe("raise", min(200, int(opp_bet * 2.4)), req)
            if level >= 3 and E >= 0.63:
                return self._safe("raise", 200, req)

        if E >= pot_odds + margin and "call" in legal:
            return self._safe("call", None, req)
        return self._safe("fold", None, req)

    def _play_postflop_ml(self, req):
        street = req.get("street", "flop")
        street_idx = {"flop": 1, "turn": 2, "river": 3}.get(street, 1)
        legal = req.get("legal_actions", [])
        to_call = req.get("to_call", 0)
        pot = req.get("pot", 3)
        my_bet = req.get("my_bet", 0)
        opp_bet = req.get("opp_bet", 0)
        history = req.get("history", [])
        board = [CARD_TO_INT[c] for c in req.get("board", [])]
        hole = [CARD_TO_INT[c] for c in self.hole_cards]

        opp_r = sum(1 for h in history if h.get("street") == street and h.get("actor") == "opp" and h.get("action") == "raise")
        my_r = sum(1 for h in history if h.get("street") == street and h.get("actor") == "me" and h.get("action") == "raise")
        opp_checked = any(h.get("street") == street and h.get("actor") == "opp" and h.get("action") == "check" for h in history)
        is_3bet_pot = sum(1 for h in history if h.get("street") == "preflop" and h.get("action") == "raise") >= 2
        prior_opp_r = sum(1 for h in history if h.get("street") not in (street, "preflop") and h.get("actor") == "opp" and h.get("action") == "raise")

        k = self._agg_k(history, street, opp_bet, my_bet, pot)
        pre_f = self._opp_pre_range(history)
        self.pre_f = pre_f
        kc = 0.42 if (opp_checked and to_call == 0) else 0.0

        if street == "flop":
            self.flop_opp_raises = max(self.flop_opp_raises, opp_r)
            if opp_r > 0 and opp_bet > 0:
                self.flop_bet_frac = opp_bet / max(2.0, pot - opp_bet - my_bet)

        dead_extra = [CARD_TO_INT[d] for d in (self.my_discard, self.opp_discard) if d in CARD_TO_INT]
        if street == "flop" and len(hole) == 3:
            dl = self._deadline(0.20)
            eqs = flop_equities_ml(hole, board[:3], [], k, dl, min_samples=self._floor(), max_samples=2400, pre_f=pre_f)
            raw_equity = max(eqs)
            hand_cat = 1
        else:
            weight = 0.22 if street == "turn" else 0.12
            dl = self._deadline(weight)
            opp_disc = CARD_TO_INT[self.opp_discard] if self.opp_discard in CARD_TO_INT else None
            raw_equity = post_equity_ml(
                hole, board, dead_extra, opp_disc, k, kc, self.opp.inconsistent_weight, dl, pre_f, self._floor() * 3
            )
            hand_cat = eval_board_hand(hole[0], hole[1], board) >> 20

        equity = raw_equity
        if to_call > 0:
            if street == "river":
                equity += self._river_cal_bias()
            elif street == "turn":
                equity += 0.5 * self._river_cal_bias()

            # v2 Paired-board underpair & multi-barrel safeguard:
            # On a paired board (`len(set(board_ranks)) < len(board)`), a pocket pair below the top board rank
            # evaluates as `cat == 2` (two pair), inflating its raw equity against random hands when facing multi-street barrels
            board_ranks = [CARD_RANK[c] for c in board]
            if len(hole) == 2 and len(set(board_ranks)) < len(board_ranks):
                max_br = max(board_ranks)
                if CARD_RANK[hole[0]] < max_br and CARD_RANK[hole[1]] < max_br and hand_cat <= 2:
                    if prior_opp_r >= 1 or is_3bet_pot:
                        equity -= 0.08

        danger = board_danger_int(board)
        opp_hi_disc = bool(self.opp_discard in CARD_TO_INT and CARD_RANK[CARD_TO_INT[self.opp_discard]] >= 9)
        pot_odds = to_call / max(1.0, float(pot + to_call))

        # ------------------------------------------------------------------
        # Scenario 1: Checked to us (`to_call == 0`) — Quantile-Conditioned Multi-Sizing EV Grid
        # ------------------------------------------------------------------
        if to_call == 0:
            if "raise" not in legal:
                return self._safe("check", None, req)

            ev_check = equity * pot
            best_action = self._safe("check", None, req)
            best_ev = ev_check + 0.40
            best_feats = None

            lo_r, hi_r = int(req.get("min_raise", 2)), int(req.get("max_raise", 200))
            # v2 Sizing Grid: Cap candidate bet sizes by hand strength so one-pair hands don't 1.15x overbet 3 streets
            if equity >= 0.84:
                cand_fracs = (0.55, 0.80, 1.10)
            elif equity >= 0.70:
                cand_fracs = (0.45, 0.65)
            elif equity >= 0.60:
                cand_fracs = (0.40, 0.52)
            else:
                cand_fracs = (0.45, 0.60)

            cand_targets = {max(lo_r, min(hi_r, my_bet + max(2, int(round(pot * frac))))) for frac in cand_fracs}
            if equity >= 0.88 and (street == "river" or hi_r <= 2.2 * pot):
                cand_targets.add(hi_r)

            for target in sorted(cand_targets):
                B = target - my_bet
                if B <= 0:
                    continue
                bet_to_pot = B / max(1.0, float(pot))
                x = self.nn.build_features(
                    street_idx, B, pot, danger, self.opp.fold_to_bet, self.opp.agg,
                    self.opp.vpip, is_3bet_pot, opp_checked, opp_hi_disc
                )
                p_fold, p_call, p_raise, _, _ = self.nn.predict(x)

                # v2 Quantile-Conditioned Adverse Selection:
                # When opponent folds bottom `p_fold` of their range, our equity against the continuing `1 - p_fold`
                # top quantile shrinks nonlinearly with `p_fold` and `bet_to_pot`!
                if equity >= 0.88:
                    eq_when_called = max(0.55, equity - 0.10 * p_fold)
                else:
                    quantile_cut = p_fold * (0.55 + 0.22 * min(1.5, bet_to_pot))
                    eq_when_called = max(0.02, min(0.96, (equity - quantile_cut) / max(0.15, 1.0 - quantile_cut)))

                ev_called = eq_when_called * (pot + 2.0 * B) - B
                if equity >= 0.80:
                    ev_raised = (equity - 0.08) * (pot + 4.2 * B) - 2.1 * B
                else:
                    ev_raised = -B

                ev_bet = p_fold * pot + p_call * ev_called + p_raise * ev_raised

                # Do not bet medium hands (0.24..0.60) unless opponent is a verified over-folder (`fold_to_bet >= 0.44`)
                if 0.24 <= equity < 0.61 and (self.opp.faced_bet < 6 or self.opp.fold_to_bet < 0.44 or pot > 85):
                    continue
                # On Turn/River, pure air bluffs (`equity < 0.24`) require `opp_checked` and `fold_to_bet >= 0.50`
                if equity < 0.24 and (street == "flop" or not opp_checked or self.opp.fold_to_bet < 0.50 or pot > 75):
                    continue

                if ev_bet > best_ev:
                    best_ev = ev_bet
                    best_action = self._safe("raise", target, req)
                    best_feats = x

            if best_action.get("action") == "raise" and best_feats is not None:
                self.pending_bet_features = best_feats
            return best_action

        # ------------------------------------------------------------------
        # Scenario 2: Facing a Bet (`to_call > 0`) — Bayesian Call EV + ML Raise EV
        # ------------------------------------------------------------------
        margin = 0.030 + (0.015 if street == "river" else 0.0) + 0.025 * max(0, opp_r - 1)
        if to_call >= req.get("my_stack", 200) - 0.5:
            margin = 0.015

        # v2 Value/Bluff Raise Evaluator:
        # Never re-raise (`my_r >= 1`) without near-nut equity (`equity >= 0.88`), preventing baby-flush 3-bet punts!
        rt = 0.82 + (0.03 if street == "river" else 0.0) + (0.07 if (my_r >= 1 or opp_r >= 2) else 0.0)
        can_bluff_raise = (
            opp_r == 1
            and my_r == 0
            and to_call < req.get("my_stack", 200) - 0.5
            and self.opp.faced_bet >= 6
            and self.opp.fold_to_bet >= 0.45
        )
        if "raise" in legal and (equity >= rt or can_bluff_raise):
            lo_r, hi_r = int(req.get("min_raise", 2)), int(req.get("max_raise", 200))
            if equity >= rt:
                mult = 0.70 if equity < 0.90 else 1.00
                target = max(lo_r, min(hi_r, opp_bet + int(round((pot + to_call) * mult))))
                R = target - my_bet
                x = self.nn.build_features(
                    street_idx, max(2, target - opp_bet), pot + to_call, danger, self.opp.fold_to_bet,
                    self.opp.agg, self.opp.vpip, is_3bet_pot, False, opp_hi_disc
                )
                self.pending_bet_features = x
                return self._safe("raise", target, req)
            elif can_bluff_raise:
                target = max(lo_r, min(hi_r, opp_bet + int(round((pot + to_call) * 0.78))))
                R = target - my_bet
                if R <= 65:
                    x = self.nn.build_features(
                        street_idx, max(2, target - opp_bet), pot + to_call, danger, self.opp.fold_to_bet,
                        self.opp.agg, self.opp.vpip, is_3bet_pot, False, opp_hi_disc
                    )
                    p_fold, p_call, p_raise, _, _ = self.nn.predict(x)
                    e2 = max(0.0, equity - 0.22 * p_fold)
                    ev_raise = p_fold * pot + (1.0 - p_fold) * (e2 * (pot + R - to_call) - (1.0 - e2) * R)
                    ev_call = equity * pot - (1.0 - equity) * to_call if equity >= pot_odds - 0.04 else 0.0
                    if ev_raise > max(ev_call, 0.0) + 1.2 and random.random() < 0.70:
                        self.pending_bet_features = x
                        return self._safe("raise", target, req)

        if equity >= pot_odds + margin and "call" in legal:
            if street == "river":
                self.pend_river_est = raw_equity
            return self._safe("call", None, req)

        if to_call <= max(2, pot // 8) and equity >= pot_odds - 0.01 and "call" in legal:
            return self._safe("call", None, req)

        return self._safe("fold", None, req)


def _fallback_action(msg):
    legal = msg.get("legal_actions", [])
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal and msg.get("to_call", 0) <= 2:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else legal[0]}


AdaptiveBot = MLSparBot


def main():
    bot = MLSparBot()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        if "time_left_ms" in msg:
            try:
                bot.time_left_ms = float(msg["time_left_ms"])
            except Exception:
                pass
        try:
            if t == "action_request":
                reply = bot.choose_action(msg)
                print(json.dumps(reply, separators=(",", ":")), flush=True)
            elif t == "discard_request":
                reply = bot.choose_discard(msg)
                print(json.dumps(reply, separators=(",", ":")), flush=True)
            elif t == "init":
                bot.handle_init(msg)
            elif t == "new_hand":
                bot.handle_new_hand(msg)
            elif t == "discard_result":
                bot.handle_discard_result(msg)
            elif t == "hand_result":
                bot.handle_hand_result(msg)
            elif t == "match_end":
                break
        except Exception as exc:
            print(f"[SPAR_BOT_ERROR] {exc!r}", file=sys.stderr, flush=True)
            if t == "action_request":
                print(json.dumps(_fallback_action(msg)), flush=True)
            elif t == "discard_request":
                cards = msg.get("hole_cards", ["2s", "3s", "4s"])
                worst = min(cards, key=lambda c: (RANK_TO_INT.get(c[0], 0), SUIT_TO_INT.get(c[1], 0)))
                print(json.dumps({"type": "discard", "card": worst}), flush=True)


if __name__ == "__main__":
    main()
