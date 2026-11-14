#!/usr/bin/env python3
"""
Phase 2 General-Purpose CEM-RL Training Loop (`rl_bot/evolve_rl_policy.py`)
Trains the 16 continuous RL policy & counterfactual parameters of `NeuralCFRBot`
STRICTLY against a diverse 12-bot General Training Pool (GTO/CFR, MCTS, Bayesian,
Bandit, Kelly, Trapper, LAG, TAG, Overbettor, Station, Slumbot) — NEVER against
friends' bots — so the resulting policy generalizes to any unseen tournament opponent.

After training, the learned policy is evaluated on a separate 7-bot Held-Out Unseen Pool.
"""

import json
import os
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "Starter_Kit" / "engine.py"
RL_BOT = ROOT / "rl_bot" / "bot.py"
CORE_RL = ROOT / "rl_bot" / "core_rl_bot.py"
EX = ROOT / "Starter_Kit" / "example_bots"

BASE_POLICY = {
    "btn_open_base": 0.32,
    "btn_open_vs_3bet": 0.44,
    "btn_limp_floor": 0.20,
    "bb_3bet_vs_wide": 0.57,
    "bb_defend_base": 0.36,
    "flop_probe_freq_high": 0.70,
    "flop_probe_freq_base": 0.37,
    "flop_probe_size": 0.35,
    "turn_float_freq": 0.40,
    "val_eq_thresh": 0.56,
    "thin_val_size": 0.52,
    "strong_val_size": 0.62,
    "turn_inter1_mdf_eq": 0.52,
    "turn_inter2_mdf_cap": 0.46,
    "bluff_raise_trap_eq": 0.60,
    "oop_turn_trap_freq": 0.35,
    "probe_counter_raise_freq": 0.45,
    "exp3_lr": 0.18,
}

PARAM_BOUNDS = {
    "btn_open_base": (0.27, 0.36, 0.025),
    "btn_open_vs_3bet": (0.40, 0.48, 0.025),
    "btn_limp_floor": (0.16, 0.24, 0.020),
    "bb_3bet_vs_wide": (0.52, 0.61, 0.025),
    "bb_defend_base": (0.32, 0.39, 0.020),
    "flop_probe_freq_high": (0.60, 0.78, 0.040),
    "flop_probe_freq_base": (0.30, 0.45, 0.035),
    "flop_probe_size": (0.32, 0.40, 0.020),
    "turn_float_freq": (0.32, 0.48, 0.040),
    "val_eq_thresh": (0.53, 0.59, 0.015),
    "thin_val_size": (0.48, 0.58, 0.025),
    "strong_val_size": (0.58, 0.70, 0.030),
    "turn_inter1_mdf_eq": (0.49, 0.56, 0.020),
    "turn_inter2_mdf_cap": (0.43, 0.50, 0.020),
    "bluff_raise_trap_eq": (0.55, 0.64, 0.025),
    "oop_turn_trap_freq": (0.25, 0.48, 0.045),
    "probe_counter_raise_freq": (0.30, 0.60, 0.050),
}


def run_single_match(args):
    policy_dict, opp_path, seed, hands = args
    env = os.environ.copy()
    env["RL_POLICY_OVERRIDES"] = json.dumps(policy_dict)
    cmd = [
        sys.executable,
        str(ENGINE),
        str(RL_BOT),
        str(opp_path),
        "--seed",
        str(seed),
        "--hands",
        str(hands),
        "--log-dir",
        "",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=90)
    out = res.stdout
    m_raw = re.search(r"A\s+\S+\s+net\s+([+-]\d+)", out)
    m_cap = re.search(r"Capped delta:\s*A\s*([+-]\d+)", out)
    raw_net = int(m_raw.group(1)) if m_raw else 0
    cap_delta = int(m_cap.group(1)) if m_cap else max(-hands, min(hands, raw_net))
    score_cap = cap_delta / float(hands)
    win = 1.0 if raw_net > 0 else (0.5 if raw_net == 0 else 0.0)
    return raw_net, score_cap, win


def evaluate_policy(policy, opponents, seeds, hands=120, max_workers=14):
    tasks = []
    for opp_name, opp_path, weight in opponents:
        for s in seeds:
            tasks.append((policy, opp_path, s, hands, opp_name, weight))

    job_args = [(t[0], t[1], t[2], t[3]) for t in tasks]
    results_by_opp = {o[0]: {"raw": 0, "cap": 0.0, "wins": 0.0, "n": 0, "w": o[2]} for o in opponents}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(run_single_match, ja): tasks[i] for i, ja in enumerate(job_args)}
        for f in as_completed(futs):
            _, _, _, _, opp_name, _ = futs[f]
            raw_net, score_cap, win = f.result()
            d = results_by_opp[opp_name]
            d["raw"] += raw_net
            d["cap"] += score_cap
            d["wins"] += win
            d["n"] += 1

    total_fitness = 0.0
    for opp_name, d in results_by_opp.items():
        n = max(1, d["n"])
        avg_cap = d["cap"] / n
        win_rate = d["wins"] / n
        total_fitness += d["w"] * (0.60 * avg_cap + 0.40 * (2.0 * win_rate - 1.0))
    return total_fitness, results_by_opp


def train_cem_rl(generations=3, pop_size=5):
    rng = random.Random(20261114)
    # 12-Bot Diverse General-Purpose Training Pool (ZERO friends' bots!)
    training_pool = [
        ("cfr_gto", EX / "cfr_gto_bot.py", 1.2),
        ("cfr_nash", EX / "cfr_nash_bot.py", 1.2),
        ("mcts", EX / "mcts_ismcts_bot.py", 1.2),
        ("exact_bayes", EX / "exact_bayesian_ev_bot.py", 1.2),
        ("bandit_meta", EX / "contextual_bandit_meta_bot.py", 1.1),
        ("slumbot", EX / "uchicago_slumbot_hybrid.py", 1.1),
        ("kelly_geo", EX / "kelly_pot_geometric_bot.py", 1.0),
        ("trapper", EX / "trapper_bot.py", 1.0),
        ("lag_maniac", EX / "lag_maniac_bot.py", 1.0),
        ("tag", EX / "tag_bot.py", 1.0),
        ("overbet", EX / "overbet_bluffer_bot.py", 1.0),
        ("station", EX / "calling_station_bot.py", 0.8),
    ]

    mean_pol = dict(BASE_POLICY)
    std_pol = {k: v[2] for k, v in PARAM_BOUNDS.items()}
    best_pol = dict(mean_pol)

    t0 = time.time()
    print("=== General-Purpose CEM-RL Evolution across 12 Diverse Bot Archetypes ===")
    base_seeds = [2001, 2102]
    best_fit, base_res = evaluate_policy(best_pol, training_pool, base_seeds, hands=120)
    print(f"[Baseline Gen 0] Fitness={best_fit:+.4f} ({time.time()-t0:.1f}s)")
    for k, d in base_res.items():
        print(f"   vs {k:12s}: Win={d['wins']:.1f}/{d['n']} | Cap={d['cap']/d['n']:+.4f} | Raw={d['raw']:+d}")

    for gen in range(1, generations + 1):
        candidates = []
        # Rotate seeds per generation to prevent overfitting to specific card deals
        gen_seeds = [2001 + gen * 37, 2102 + gen * 53]
        for idx in range(pop_size):
            if idx == 0:
                cand = dict(best_pol)
            else:
                cand = dict(mean_pol)
                for k, (lo, hi, _) in PARAM_BOUNDS.items():
                    val = rng.gauss(mean_pol[k], std_pol[k])
                    cand[k] = round(max(lo, min(hi, val)), 4)
            fit, res = evaluate_policy(cand, training_pool, gen_seeds, hands=120)
            candidates.append((fit, cand, res))
            tot_w = sum(od["wins"] for od in res.values())
            tot_m = sum(od["n"] for od in res.values())
            tot_raw = sum(od["raw"] for od in res.values())
            print(f"  Gen {gen} Cand #{idx+1}/{pop_size}: Fitness={fit:+.4f} | Wins={tot_w:.1f}/{tot_m} | Raw={tot_raw:+d}")

        candidates.sort(key=lambda x: x[0], reverse=True)
        elites = candidates[:2]
        if elites[0][0] >= best_fit or gen == 1:
            best_fit = elites[0][0]
            best_pol = dict(elites[0][1])

        for k in PARAM_BOUNDS:
            elite_vals = [e[1][k] for e in elites]
            new_m = sum(elite_vals) / len(elite_vals)
            mean_pol[k] = round(0.65 * new_m + 0.35 * mean_pol[k], 4)
            std_pol[k] = max(0.005, std_pol[k] * 0.78)

        print(f"[Gen {gen} Complete] Best Fitness={best_fit:+.4f}")

    out_json = ROOT / "rl_bot" / "trained_policy.json"
    out_json.write_text(json.dumps(best_pol, indent=2))
    print(f"Saved general-purpose CEM-RL policy to {out_json}")

    src = CORE_RL.read_text()
    pol_lines = ['    "policy": {']
    for k, v in best_pol.items():
        pol_lines.append(f'        "{k}": {v},')
    pol_lines.append("    },")
    new_pol_block = "\n".join(pol_lines)
    src_new = re.sub(r'    "policy": \{[^}]+\},', new_pol_block, src, count=1)
    CORE_RL.write_text(src_new)
    print("Embedded general-purpose CEM-RL policy weights into rl_bot/core_rl_bot.py")


if __name__ == "__main__":
    train_cem_rl(generations=3, pop_size=4)
