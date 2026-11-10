#!/usr/bin/env python3
"""
Pokerbots Official Two-Leg Tournament & Practice Match Runner (Discard Hold'em, Protocol 1.0).
Built directly on top of the organisers' updated two-leg `engine.py` (Rulebook Sections 7 & 8).

Supports three modes:
1. Head-to-Head Practice Series (default when 2 bots are passed):
     python3 run_tournament.py bot.py /path/to/friend_bot.py [--stage 1|2] [--matches 5] [--hands 400]
2. Full Round-Robin Tournament (all-pairs, like Stage 2 Finals or <64-team Stage 1):
     python3 run_tournament.py --round-robin bot.py /path/to/friend_bot.py example_bots/cfr_nash_bot.py ...
3. Swiss System Qualifier Tournament (like Stage 1 Swiss, Rulebook 7.2-7.4):
     python3 run_tournament.py --swiss --rounds 6 bot.py /path/to/friend_bot.py ...
"""

import argparse
import os
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(BASE_DIR, "engine.py")
STAGES = {
    1: {"hands": 400, "time_bank_s": 30, "cap": 400},
    2: {"hands": 1000, "time_bank_s": 75, "cap": 1000},
}


def short_name(path):
    base = os.path.basename(os.path.abspath(path))
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if base == "bot.py" and parent and parent != "Starter_Kit":
        return f"{parent}/{base}"
    return base


def play_two_leg_match(bot_a, bot_b, stage=1, hands=None, seed=None, log_dir="", python_exe=sys.executable):
    """
    Runs a single official two-leg match via `engine.py` and parses the official result,
    including raw net chips, capped delta (Rulebook 8.3), time used, notices, and forfeits.
    """
    st = STAGES[stage]
    match_hands = hands or st["hands"]
    cap = st["cap"]
    seed_str = str(seed if seed is not None else random.SystemRandom().randrange(10 ** 9))

    cmd = [
        python_exe,
        ENGINE,
        os.path.abspath(bot_a),
        os.path.abspath(bot_b),
        "--stage",
        str(stage),
        "--hands",
        str(match_hands),
        "--seed",
        seed_str,
        "--log-dir",
        log_dir,
    ]
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    out = res.stdout

    # Parse lines:
    #   A  bot.py                           net    +312   time used   0.42 s (bank 30 s per leg)   notices 0
    m_a = re.search(r"^\s*A\s+\S+\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+) s.*notices (\d+)", out, re.M)
    m_b = re.search(r"^\s*B\s+\S+\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+) s.*notices (\d+)", out, re.M)
    m_played = re.search(r"Seed \S+ \| (\d+) of (\d+) hands played", out)

    if not m_a or not m_b:
        return {
            "error": out + "\n" + res.stderr,
            "seed": seed_str,
            "hands": match_hands,
            "net_a": 0,
            "net_b": 0,
            "capped_a": 0,
            "capped_b": 0,
            "pts_a": 0.5,
            "pts_b": 0.5,
            "time_a": 0.0,
            "time_b": 0.0,
            "notices_a": 0,
            "notices_b": 0,
            "status": "ERROR",
        }

    net_a = int(m_a.group(1))
    time_a = float(m_a.group(2))
    notices_a = int(m_a.group(3))
    net_b = int(m_b.group(1))
    time_b = float(m_b.group(2))
    notices_b = int(m_b.group(3))
    played = int(m_played.group(1)) if m_played else match_hands

    # Check forfeit status (Rulebook 9.2)
    if "Result: A wins by forfeit" in out:
        pts_a, pts_b = 1.0, 0.0
        capped_a, capped_b = cap, -cap
        status = "B_FORFEIT"
    elif "Result: B wins by forfeit" in out:
        pts_a, pts_b = 0.0, 1.0
        capped_a, capped_b = -cap, cap
        status = "A_FORFEIT"
    elif "both forfeited" in out:
        pts_a, pts_b = 0.5, 0.5
        capped_a, capped_b = 0, 0
        status = "BOTH_FORFEIT"
    else:
        capped_a = max(-cap, min(cap, net_a))
        capped_b = -capped_a
        if net_a > net_b:
            pts_a, pts_b = 1.0, 0.0
            status = "A_WIN"
        elif net_b > net_a:
            pts_a, pts_b = 0.0, 1.0
            status = "B_WIN"
        else:
            pts_a, pts_b = 0.5, 0.5
            status = "DRAW"

    return {
        "seed": seed_str,
        "hands": match_hands,
        "played": played,
        "net_a": net_a,
        "net_b": net_b,
        "capped_a": capped_a,
        "capped_b": capped_b,
        "pts_a": pts_a,
        "pts_b": pts_b,
        "time_a": time_a,
        "time_b": time_b,
        "notices_a": notices_a,
        "notices_b": notices_b,
        "status": status,
        "raw_output": out,
    }


def run_head_to_head_series(bot_a, bot_b, stage=1, matches=5, hands=None, seeds=None, parallel=4, log_prefix=""):
    st = STAGES[stage]
    match_hands = hands or st["hands"]
    cap = st["cap"]
    if not seeds:
        seeds = [1001 + i * 101 for i in range(matches)]
    else:
        matches = len(seeds)

    name_a = short_name(bot_a)
    name_b = short_name(bot_b)

    print(f"\n================================================================================================")
    print(f"  HEAD-TO-HEAD PRACTICE SERIES: {name_a} (A) vs {name_b} (B)")
    print(f"  Format: Stage {stage} | {matches} Matches x {match_hands} Hands (2 Legs of {match_hands // 2} Deals) | Cap C = {cap}")
    print(f"================================================================================================")

    futures = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for idx, s in enumerate(seeds):
            ldir = f"{log_prefix}_match_{idx + 1}_seed_{s}" if log_prefix else ""
            futures.append((idx + 1, s, ex.submit(play_two_leg_match, bot_a, bot_b, stage, match_hands, s, ldir)))

    results = []
    print(f"{'Match':<6} | {'Seed':<8} | {'Winner':<14} | {'Raw Net (A)':<12} | {'Capped (A)':<11} | {'Time A / B (2 Legs)':<20} | {'Notices A/B'}")
    print("-" * 96)

    for idx, s, fut in futures:
        r = fut.result()
        results.append(r)
        if r["pts_a"] == 1.0:
            win_str = f"A ({name_a[:8]})"
        elif r["pts_b"] == 1.0:
            win_str = f"B ({name_b[:8]})"
        else:
            win_str = "DRAW"
        if "FORFEIT" in r["status"]:
            win_str += "*"
        t_str = f"{r['time_a']:5.2f}s / {r['time_b']:5.2f}s"
        n_str = f"{r['notices_a']} / {r['notices_b']}"
        print(f"#{idx:<5} | {r['seed']:<8} | {win_str:<14} | {r['net_a']:+12d} | {r['capped_a']:+11d} | {t_str:<20} | {n_str}")

    print("-" * 96)
    w = sum(1 for r in results if r["pts_a"] == 1.0)
    d = sum(1 for r in results if r["pts_a"] == 0.5)
    l = sum(1 for r in results if r["pts_a"] == 0.0)
    tot_pts_a = sum(r["pts_a"] for r in results)
    tot_pts_b = sum(r["pts_b"] for r in results)
    tot_net_a = sum(r["net_a"] for r in results)
    tot_cap_a = sum(r["capped_a"] for r in results)
    tot_hands = sum(r["hands"] for r in results)
    score_cap_a = tot_cap_a / max(1, tot_hands)
    raw_per_hand_a = tot_net_a / max(1, tot_hands)

    print(f"SERIES SUMMARY ({tot_hands} Total Hands across {matches} Two-Leg Matches):")
    print(f"  A ({name_a:<22}): Match Points = {tot_pts_a:4.1f} / {matches} ({w}W-{d}D-{l}L) | "
          f"Score_cap = {score_cap_a:+.4f}/hand | Raw Net = {tot_net_a:+d} ({raw_per_hand_a:+.3f}/hand)")
    print(f"  B ({name_b:<22}): Match Points = {tot_pts_b:4.1f} / {matches} ({l}W-{d}D-{w}L) | "
          f"Score_cap = {-score_cap_a:+.4f}/hand | Raw Net = {-tot_net_a:+d} ({-raw_per_hand_a:+.3f}/hand)")
    print(f"================================================================================================\n")
    return results


def run_round_robin(bots, stage=1, hands=None, seeds=(101, 202), parallel=4):
    st = STAGES[stage]
    match_hands = hands or st["hands"]
    cap = st["cap"]
    n = len(bots)
    names = [short_name(b) for b in bots]

    print(f"\n================================================================================================")
    print(f"  OFFICIAL ROUND-ROBIN TOURNAMENT ({n} Bots, {len(seeds)} Seeds/Pair, Stage {stage}: {match_hands} Hands/Match)")
    print(f"================================================================================================")

    pairings = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for i in range(n):
            for j in range(i + 1, n):
                for s in seeds:
                    fut = ex.submit(play_two_leg_match, bots[i], bots[j], stage, match_hands, s, "")
                    pairings.append((i, j, s, fut))

    # Standings accumulator: [pts, w, d, l, capped_sum, raw_sum, hands_scheduled, time_used, notices]
    stats = {i: [0.0, 0, 0, 0, 0, 0, 0, 0.0, 0] for i in range(n)}

    for i, j, s, fut in pairings:
        r = fut.result()
        # Update Bot i (A)
        stats[i][0] += r["pts_a"]
        stats[i][1] += 1 if r["pts_a"] == 1.0 else 0
        stats[i][2] += 1 if r["pts_a"] == 0.5 else 0
        stats[i][3] += 1 if r["pts_a"] == 0.0 else 0
        stats[i][4] += r["capped_a"]
        stats[i][5] += r["net_a"]
        stats[i][6] += r["hands"]
        stats[i][7] += r["time_a"]
        stats[i][8] += r["notices_a"]

        # Update Bot j (B)
        stats[j][0] += r["pts_b"]
        stats[j][1] += 1 if r["pts_b"] == 1.0 else 0
        stats[j][2] += 1 if r["pts_b"] == 0.5 else 0
        stats[j][3] += 1 if r["pts_b"] == 0.0 else 0
        stats[j][4] += r["capped_b"]
        stats[j][5] += r["net_b"]
        stats[j][6] += r["hands"]
        stats[j][7] += r["time_b"]
        stats[j][8] += r["notices_b"]

    # Sort by Rulebook 8.2 & 8.3: Match Points descending, then Score_cap descending, then Raw Net descending
    ranked = sorted(
        range(n),
        key=lambda i: (stats[i][0], stats[i][4] / max(1, stats[i][6]), stats[i][5]),
        reverse=True,
    )

    print(f"{'Rank':<5} | {'Bot Name':<28} | {'Match Pts':<9} | {'W-D-L':<9} | {'Score_cap':<10} | {'Raw Net':<10} | {'Avg/Hand':<9} | {'Notices'}")
    print("-" * 105)
    for rk, idx in enumerate(ranked, 1):
        st_i = stats[idx]
        mp = st_i[0]
        wdl = f"{st_i[1]}-{st_i[2]}-{st_i[3]}"
        sc_cap = st_i[4] / max(1, st_i[6])
        raw_net = st_i[5]
        raw_avg = raw_net / max(1, st_i[6])
        print(
            f"#{rk:<4} | {names[idx]:<28} | {mp:9.1f} | {wdl:<9} | {sc_cap:+10.4f} | {raw_net:+10d} | {raw_avg:+9.3f} | {st_i[8]}"
        )
    print("================================================================================================\n")


def run_swiss(bots, rounds=6, stage=1, hands=None, base_seed=5000, parallel=4):
    """
    Official Stage 1 Swiss Qualifier simulator implementing Rulebook Sections 7.2-7.4 & 8:
      - Round 1 pairings are random (or seeded).
      - Later rounds rank teams by Match Points, then Score_cap, and pair each team with the
        closest-ranked team it has not yet played.
      - Odd team count awards a Bye (+1.0 Match Point, +0 chips, +0 hands to Score_cap) to the
        lowest-ranked team that has not yet had a Bye.
    """
    st = STAGES[stage]
    match_hands = hands or st["hands"]
    n = len(bots)
    names = [short_name(b) for b in bots]
    max_possible_rounds = n - 1 if n % 2 == 0 else n
    rounds = min(rounds, max_possible_rounds)

    print(f"\n================================================================================================")
    print(f"  OFFICIAL SWISS QUALIFIER TOURNAMENT ({n} Bots, {rounds} Rounds, Stage {stage}: {match_hands} Hands/Match)")
    print(f"================================================================================================")

    # [pts, w, d, l, capped_sum, raw_sum, hands_scheduled, had_bye]
    stats = {i: [0.0, 0, 0, 0, 0, 0, 0, False] for i in range(n)}
    played_pairs = set()
    rng = random.Random(base_seed)

    for r_no in range(1, rounds + 1):
        if r_no == 1:
            order = list(range(n))
            rng.shuffle(order)
        else:
            order = sorted(
                range(n),
                key=lambda i: (stats[i][0], stats[i][4] / max(1, stats[i][6]), rng.random()),
                reverse=True,
            )

        unpaired = list(order)
        bye_bot = None
        if len(unpaired) % 2 == 1:
            # Lowest-ranked team that has not yet had a bye (Rulebook 7.4)
            for cand in reversed(unpaired):
                if not stats[cand][7]:
                    bye_bot = cand
                    break
            if bye_bot is None:
                bye_bot = unpaired[-1]
            unpaired.remove(bye_bot)
            stats[bye_bot][0] += 1.0
            stats[bye_bot][1] += 1
            stats[bye_bot][7] = True

        round_pairs = []
        while unpaired:
            a = unpaired.pop(0)
            chosen_b = None
            for b in unpaired:
                if (min(a, b), max(a, b)) not in played_pairs:
                    chosen_b = b
                    break
            if chosen_b is None:
                chosen_b = unpaired[0]
            unpaired.remove(chosen_b)
            played_pairs.add((min(a, chosen_b), max(a, chosen_b)))
            round_pairs.append((a, chosen_b))

        futs = []
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            for pair_idx, (a, b) in enumerate(round_pairs):
                s = base_seed + r_no * 100 + pair_idx
                futs.append((a, b, ex.submit(play_two_leg_match, bots[a], bots[b], stage, match_hands, s, "")))

        print(f"--- Round {r_no} ---")
        for a, b, fut in futs:
            res = fut.result()
            stats[a][0] += res["pts_a"]
            stats[a][1] += 1 if res["pts_a"] == 1.0 else 0
            stats[a][2] += 1 if res["pts_a"] == 0.5 else 0
            stats[a][3] += 1 if res["pts_a"] == 0.0 else 0
            stats[a][4] += res["capped_a"]
            stats[a][5] += res["net_a"]
            stats[a][6] += res["hands"]

            stats[b][0] += res["pts_b"]
            stats[b][1] += 1 if res["pts_b"] == 1.0 else 0
            stats[b][2] += 1 if res["pts_b"] == 0.5 else 0
            stats[b][3] += 1 if res["pts_b"] == 0.0 else 0
            stats[b][4] += res["capped_b"]
            stats[b][5] += res["net_b"]
            stats[b][6] += res["hands"]
            print(f"  {names[a]:<24} vs {names[b]:<24} -> {res['net_a']:+6d} (Capped {res['capped_a']:+5d})")
        if bye_bot is not None:
            print(f"  {names[bye_bot]:<24} receives BYE (+1.0 Match Point)")

    ranked = sorted(
        range(n),
        key=lambda i: (stats[i][0], stats[i][4] / max(1, stats[i][6]), stats[i][5]),
        reverse=True,
    )
    print(f"\nFINAL SWISS STANDINGS AFTER {rounds} ROUNDS:")
    print(f"{'Rank':<5} | {'Bot Name':<28} | {'Match Pts':<9} | {'W-D-L':<9} | {'Score_cap':<10} | {'Raw Net':<10}")
    print("-" * 85)
    for rk, idx in enumerate(ranked, 1):
        st_i = stats[idx]
        sc_cap = st_i[4] / max(1, st_i[6])
        wdl = f"{st_i[1]}-{st_i[2]}-{st_i[3]}"
        print(f"#{rk:<4} | {names[idx]:<28} | {st_i[0]:9.1f} | {wdl:<9} | {sc_cap:+10.4f} | {st_i[5]:+10d}")
    print("================================================================================================\n")


def main():
    p = argparse.ArgumentParser(description="Official Two-Leg Tournament & Practice Match Runner (Discard Hold'em)")
    p.add_argument("bots", nargs="+", help="Paths to 2 or more bot.py files")
    p.add_argument("--stage", type=int, choices=(1, 2), default=1, help="Stage 1 (400 hands, C=400) or Stage 2 (1000 hands, C=1000)")
    p.add_argument("--hands", type=int, help="Override total hands per match (must be even)")
    p.add_argument("--matches", type=int, default=5, help="Number of matches in a 2-bot Head-to-Head series (default: 5)")
    p.add_argument("--seeds", nargs="+", help="Explicit list of seeds to run")
    p.add_argument("--round-robin", action="store_true", help="Run a Full Round-Robin tournament across all specified bots")
    p.add_argument("--swiss", action="store_true", help="Run a Stage 1 Swiss tournament across all specified bots")
    p.add_argument("--rounds", type=int, default=5, help="Number of Swiss rounds (default: 5)")
    p.add_argument("--parallel", type=int, default=4, help="Max concurrent matches (default: 4)")
    p.add_argument("--log-prefix", default="", help="Optional directory prefix to save match logs for each match")
    args = p.parse_args()

    for b in args.bots:
        if not os.path.isfile(b):
            sys.exit(f"Bot file not found: {b}")

    if args.swiss:
        if len(args.bots) < 2:
            sys.exit("Need at least 2 bots for Swiss tournament.")
        run_swiss(args.bots, rounds=args.rounds, stage=args.stage, hands=args.hands, parallel=args.parallel)
    elif args.round_robin or len(args.bots) > 2:
        seeds = args.seeds if args.seeds else [101 + 100 * i for i in range(max(1, args.matches // 2))]
        run_round_robin(args.bots, stage=args.stage, hands=args.hands, seeds=seeds, parallel=args.parallel)
    else:
        if len(args.bots) != 2:
            sys.exit("Pass exactly 2 bots for a Head-to-Head Practice Series, or use --round-robin / --swiss.")
        run_head_to_head_series(
            args.bots[0],
            args.bots[1],
            stage=args.stage,
            matches=args.matches,
            hands=args.hands,
            seeds=args.seeds,
            parallel=args.parallel,
            log_prefix=args.log_prefix,
        )


if __name__ == "__main__":
    main()
