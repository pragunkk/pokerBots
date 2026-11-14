#!/usr/bin/env python3
"""
Phase 2 RL Training Loop: Cross-Entropy Method Reinforcement Learning (CEM-RL)
Optimizes the 16 continuous RL policy & counterfactual thresholds of `NeuralCFRBot`
by playing parallel duplicate matches against:
  - `Starter_Kit/bot.py` (`AdaptiveBot v9`)
  - `opponentBots/bot_Varshith.py`
  - `spar/spar_bot.py` (`MLSparBot v2`)
  - `opponentBots/bot_Bhavesh.py`
and then embeds the winning RL policy parameters directly into `rl_bot/core_rl_bot.py`
and compiles the final standalone single-file `rl_bot/bot.py`.
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

BASE_POLICY = {
    "btn_open_base": 0.31,
    "btn_open_vs_3bet": 0.43,
    "btn_limp_floor": 0.18,
    "bb_3bet_vs_wide": 0.56,
    "bb_defend_base": 0.35,
    "flop_probe_freq_high": 0.72,
    "flop_probe_freq_base": 0.38,
    "flop_probe_size": 0.35,
    "turn_float_freq": 0.42,
    "val_eq_thresh": 0.55,
    "thin_val_size": 0.54,
    "strong_val_size": 0.66,
    "turn_inter1_mdf_eq": 0.51,
    "turn_inter2_mdf_cap": 0.46,
    "bluff_raise_trap_eq": 0.58,
    "oop_turn_trap_freq": 0.38,
    "cfr_blend": 0.32,
    "exp3_lr": 0.18,
}

PARAM_BOUNDS = {
    "btn_open_base": (0.26, 0.36, 0.025),
    "btn_open_vs_3bet": (0.39, 0.47, 0.025),
    "btn_limp_floor": (0.14, 0.22, 0.020),
    "bb_3bet_vs_wide": (0.52, 0.60, 0.025),
    "bb_defend_base": (0.32, 0.38, 0.020),
    "flop_probe_freq_high": (0.62, 0.80, 0.040),
    "flop_probe_freq_base": (0.32, 0.46, 0.035),
    "flop_probe_size": (0.32, 0.40, 0.020),
    "turn_float_freq": (0.34, 0.50, 0.040),
    "val_eq_thresh": (0.53, 0.58, 0.015),
    "thin_val_size": (0.48, 0.58, 0.025),
    "strong_val_size": (0.60, 0.72, 0.030),
    "turn_inter1_mdf_eq": (0.48, 0.55, 0.020),
    "turn_inter2_mdf_cap": (0.43, 0.50, 0.020),
    "bluff_raise_trap_eq": (0.54, 0.63, 0.025),
    "oop_turn_trap_freq": (0.25, 0.48, 0.045),
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


def evaluate_policy(policy, opponents, seeds, hands=160, max_workers=10):
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
        total_fitness += d["w"] * (0.55 * avg_cap + 0.45 * (2.0 * win_rate - 1.0))
    return total_fitness, results_by_opp


def train_cem_rl(generations=2, pop_size=5):
    rng = random.Random(20261114)
    opponents = [
        ("Varshith", ROOT / "opponentBots" / "bot_Varshith.py", 1.40),
        ("SparBot", ROOT / "spar" / "spar_bot.py", 1.25),
        ("Adaptive_v9", ROOT / "Starter_Kit" / "bot.py", 1.25),
        ("Bhavesh", ROOT / "opponentBots" / "bot_Bhavesh.py", 0.50),
    ]
    mean_pol = dict(BASE_POLICY)
    std_pol = {k: v[2] for k, v in PARAM_BOUNDS.items()}

    best_pol = dict(mean_pol)
    t0 = time.time()
    print("=== Starting CEM-RL Policy Evolution against Full Gauntlet ===")
    base_seeds = [1001, 1102, 1203]
    best_fit, base_res = evaluate_policy(best_pol, opponents, base_seeds, hands=160)
    print(f"[Baseline Gen 0] Fitness={best_fit:+.4f} ({time.time()-t0:.1f}s)")
    for k, d in base_res.items():
        print(f"   vs {k:12s}: Win={d['wins']}/{d['n']} | Score_cap={d['cap']/d['n']:+.4f} | Raw={d['raw']:+d}")

    for gen in range(1, generations + 1):
        candidates = []
        for idx in range(pop_size):
            if idx == 0:
                cand = dict(best_pol)
            else:
                cand = dict(mean_pol)
                for k, (lo, hi, _) in PARAM_BOUNDS.items():
                    val = rng.gauss(mean_pol[k], std_pol[k])
                    cand[k] = round(max(lo, min(hi, val)), 4)
            fit, res = evaluate_policy(cand, opponents, base_seeds, hands=160)
            candidates.append((fit, cand, res))
            print(f"  Gen {gen} Cand #{idx+1}/{pop_size}: Fitness={fit:+.4f} | " +
                  " | ".join(f"{ok}:{od['raw']:+d}({od['wins']:.1f}W)" for ok, od in res.items()))

        candidates.sort(key=lambda x: x[0], reverse=True)
        elites = candidates[:2]
        if elites[0][0] >= best_fit:
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
    print(f"Saved CEM-RL policy to {out_json}")

    src = CORE_RL.read_text()
    pol_lines = ['    "policy": {']
    for k, v in best_pol.items():
        pol_lines.append(f'        "{k}": {v},')
    pol_lines.append("    },")
    new_pol_block = "\n".join(pol_lines)
    src_new = re.sub(r'    "policy": \{[^}]+\},', new_pol_block, src, count=1)
    CORE_RL.write_text(src_new)
    print("Embedded trained CEM-RL policy weights into rl_bot/core_rl_bot.py")


if __name__ == "__main__":
    train_cem_rl(generations=2, pop_size=4)
