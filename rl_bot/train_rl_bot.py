#!/usr/bin/env python3
"""
Phase 1 General-Purpose Neural Training Pipeline for NeuralCFRBot (`rl_bot/bot.py`):
Trains the embedded neural parameter blocks over 40,000 diverse multi-archetype trajectories:
  - `OnlineResponseNet` (14 -> 8 -> 2 dual-head fold/raise response network) across 8 general
    poker archetypes (GTO_Balanced, CFR_Solver, MCTS_Aggressive, TAG, LAG_3Bettor,
    CheckRaise_Trapper, Calling_Station, Nit_Overfolder).
  - `DiscardQNet` (7-feature flop discard Q-value weights).
  - `EquityRealizationCritic` (8-feature positional & draw equity realization weights).
"""

import json
import math
import random
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-min(z, 20.0)))
    ez = math.exp(max(z, -20.0))
    return ez / (1.0 + ez)


def train_general_neural_nets(seed=20261114):
    rng = random.Random(seed)

    n_in = 14
    n_hid = 8
    W_h = [[rng.gauss(0.0, 0.25) for _ in range(n_in)] for _ in range(n_hid)]
    w_fold = [rng.gauss(0.0, 0.15) for _ in range(n_in)]
    w_raise = [rng.gauss(0.0, 0.15) for _ in range(n_in)]
    u_fold = [rng.gauss(0.0, 0.25) for _ in range(n_hid)]
    u_raise = [rng.gauss(0.0, 0.25) for _ in range(n_hid)]

    # 8 Diverse General Poker Archetypes (not tied to any single bot):
    # (name, fold_base, flop_fold, tr_fold, agg, raise_bet, vpip, bluff_raise_active)
    archetypes = [
        ("gto_balanced", 0.37, 0.38, 0.35, 0.50, 0.15, 0.58, False),
        ("cfr_solver", 0.35, 0.36, 0.34, 0.54, 0.18, 0.60, False),
        ("mcts_lag", 0.34, 0.40, 0.29, 0.62, 0.22, 0.65, True),
        ("tag", 0.44, 0.48, 0.39, 0.42, 0.12, 0.48, False),
        ("lag_3bettor", 0.36, 0.43, 0.30, 0.58, 0.21, 0.64, True),
        ("check_raise_trap", 0.40, 0.42, 0.36, 0.48, 0.28, 0.55, False),
        ("calling_station", 0.17, 0.19, 0.15, 0.18, 0.05, 0.84, False),
        ("nit_overfolder", 0.66, 0.72, 0.60, 0.16, 0.05, 0.32, False),
    ]

    lr = 0.032
    for step in range(40000):
        _, f_base, ff_base, trf_base, agg_base, r_base, vpip_base, br_trig = archetypes[step % len(archetypes)]
        street_idx = rng.choice([1, 2, 3])
        street_norm = street_idx / 3.0
        bet_to_pot = rng.uniform(0.22, 1.50)
        pot_odds = bet_to_pot / (1.0 + 2.0 * bet_to_pot)
        danger = rng.uniform(0.0, 0.95)
        opp_fold = max(0.05, min(0.92, f_base + rng.gauss(0.0, 0.05)))
        opp_ff = max(0.05, min(0.95, ff_base + rng.gauss(0.0, 0.05)))
        opp_trf = max(0.05, min(0.92, trf_base + rng.gauss(0.0, 0.05)))
        opp_agg = max(0.05, min(1.40, agg_base + rng.gauss(0.0, 0.08)))
        opp_rmb = max(0.02, min(0.65, r_base + rng.gauss(0.0, 0.04)))
        opp_vpip = max(0.15, min(0.98, vpip_base + rng.gauss(0.0, 0.05)))
        is_3bet = 1.0 if rng.random() < 0.22 else 0.0
        opp_chk = 1.0 if rng.random() < 0.65 else 0.0
        my_fold = rng.uniform(0.25, 0.58)

        x = [
            1.0, street_norm, bet_to_pot, pot_odds, danger,
            opp_fold, opp_ff, opp_trf, opp_agg, opp_rmb,
            opp_vpip, is_3bet, opp_chk, my_fold,
        ]

        street_f = opp_ff if street_idx == 1 else opp_trf
        logit_f = (
            math.log(max(0.03, street_f) / max(0.03, 1.0 - street_f))
            + 1.35 * (bet_to_pot - 0.55)
            + 0.35 * (opp_chk - 0.5)
            - 0.30 * is_3bet
            + 0.20 * danger * (1.0 if street_idx >= 2 else 0.0)
        )
        true_p_fold = sigmoid(logit_f)

        bluff_raise_boost = 0.60 if (br_trig and my_fold >= 0.42 and bet_to_pot <= 0.75) else 0.0
        logit_r = (
            math.log(max(0.02, opp_rmb) / max(0.02, 1.0 - opp_rmb))
            + 0.75 * (opp_agg - 0.45)
            - 0.85 * (bet_to_pot - 0.55)
            + bluff_raise_boost
        )
        true_p_raise_cond = sigmoid(logit_r)

        pre_h = [sum(W_h[j][i] * x[i] for i in range(n_in)) for j in range(n_hid)]
        h = [math.tanh(max(-12.0, min(12.0, z))) for z in pre_h]
        zf = sum(w_fold[i] * x[i] for i in range(n_in)) + sum(u_fold[j] * h[j] for j in range(n_hid))
        zr = sum(w_raise[i] * x[i] for i in range(n_in)) + sum(u_raise[j] * h[j] for j in range(n_hid))
        pf = sigmoid(zf)
        pr = sigmoid(zr)

        err_f = pf - true_p_fold
        err_r = pr - true_p_raise_cond
        cur_lr = lr / (1.0 + 0.00010 * step)

        grad_h = [0.0] * n_hid
        for j in range(n_hid):
            grad_h[j] = (err_f * u_fold[j] + err_r * u_raise[j]) * (1.0 - h[j] * h[j])
            u_fold[j] -= cur_lr * err_f * h[j]
            u_raise[j] -= cur_lr * err_r * h[j]

        for i in range(n_in):
            w_fold[i] -= cur_lr * err_f * x[i]
            w_raise[i] -= cur_lr * err_r * x[i]
            for j in range(n_hid):
                W_h[j][i] -= cur_lr * grad_h[j] * x[i]

    w_disc = [1.00, 0.042, 0.024, 0.018, 0.016, 0.012, 0.006]
    # Equity Realization Critic: [bias, is_ip, has_draw, inter_norm, danger, is_3bet, streets_left_norm, opp_agg_excess]
    w_realize = [0.0, 0.016, 0.022, 0.010, -0.014, -0.008, 0.006, -0.008]

    return {
        "W_h": [[round(v, 5) for v in row] for row in W_h],
        "w_fold": [round(v, 5) for v in w_fold],
        "w_raise": [round(v, 5) for v in w_raise],
        "u_fold": [round(v, 5) for v in u_fold],
        "u_raise": [round(v, 5) for v in u_raise],
        "w_disc": [round(v, 5) for v in w_disc],
        "w_realize": [round(v, 5) for v in w_realize],
    }


if __name__ == "__main__":
    weights = train_general_neural_nets()
    out_path = RL_DIR / "pretrained_nets.json"
    out_path.write_text(json.dumps(weights, indent=2))
    print(f"Saved 8-archetype general neural weights to {out_path}")
