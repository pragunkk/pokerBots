"""
Parallel Tournament Simulator & Benchmarking Harness for Discard Hold'em.
Runs multi-seed matches between our Adaptive Bot v5 and the complete 20-Bot Championship Suite:
  - 5 Extreme Heuristic Stress-Test Bots
  - 5 Proven Computer-Poker Algorithm Bots
  - 8 Archetype & Rival Meta Bots
  - 2 Self-Play Evolutionary Mirrors (Adaptive v4 Mirror & Adaptive v3 Mirror)
"""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(BASE_DIR, "engine.py")


def run_single_match(bot_a, bot_b, hands=200, seed=100):
    cmd = [
        sys.executable,
        ENGINE,
        bot_a,
        bot_b,
        "--hands",
        str(hands),
        "--seed",
        str(seed),
        "--log-dir",
        "",
    ]
    res = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    out = res.stdout
    m_a = re.search(r"^\s*A\s+\S+\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+) s.*notices (\d+)", out, re.M)
    m_b = re.search(r"^\s*B\s+\S+\s+net\s+([+-]?\d+)\s+time used\s+([\d.]+) s.*notices (\d+)", out, re.M)
    if not m_a or not m_b:
        return {"error": out + "\n" + res.stderr}
    return {
        "seed": seed,
        "hands": hands,
        "net_a": int(m_a.group(1)),
        "time_a": float(m_a.group(2)),
        "notices_a": int(m_a.group(3)),
        "net_b": int(m_b.group(1)),
        "time_b": float(m_b.group(2)),
        "notices_b": int(m_b.group(3)),
    }


def run_suite(hero_bot, opponents, hands_per_match=300, seeds=(101, 202, 303, 404), title="BENCHMARK"):
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for opp_name, opp_path in opponents:
            for seed in seeds:
                fut = ex.submit(run_single_match, hero_bot, opp_path, hands_per_match, seed)
                tasks.append((opp_name, opp_path, seed, fut))

    results = {}
    for opp_name, opp_path, seed, fut in tasks:
        r = fut.result()
        results.setdefault(opp_name, []).append(r)

    print(f"\n=== {title}: {hero_bot} ({hands_per_match} hands/match x {len(seeds)} seeds) ===")
    h1, h2, h3, h4, h5, h6 = "Opponent / Algorithm", "W-D-L", "Seed Breakdown", "Total Net", "Avg/Hand", "Notices"
    print(f"{h1:<36} | {h2:<7} | {h3:<30} | {h4:<10} | {h5:<9} | {h6}")
    print("-" * 110)
    grand_wins, grand_matches, grand_net, grand_hands = 0.0, 0, 0, 0
    for opp_name, _ in opponents:
        runs = results[opp_name]
        w = sum(1 for r in runs if r.get("net_a", 0) > 0)
        d = sum(1 for r in runs if r.get("net_a", 0) == 0)
        l = sum(1 for r in runs if r.get("net_a", 0) < 0)
        tot_net = sum(r.get("net_a", 0) for r in runs)
        tot_h = sum(r.get("hands", hands_per_match) for r in runs)
        not_a = sum(r.get("notices_a", 0) for r in runs)
        sb = ", ".join(f"{r.get('net_a', 0):+d}" for r in runs)
        print(
            f"{opp_name:<36} | {w}-{d}-{l:<3} | {sb:<30} | {tot_net:+10d} | {tot_net/tot_h:+9.3f} | {not_a}"
        )
        grand_wins += w + 0.5 * d
        grand_matches += len(runs)
        grand_net += tot_net
        grand_hands += tot_h
    print("-" * 110)
    print(
        f"OVERALL: Match Points {grand_wins:.1f}/{grand_matches} ({100*grand_wins/grand_matches:.1f}%), "
        f"Total Net {grand_net:+d} ({grand_net/grand_hands:+.3f} chips/hand)\n"
    )


EXTREME_HEURISTICS = [
    ("Always Fold Heuristic", "example_bots/always_fold_bot.py"),
    ("Always Call Heuristic", "example_bots/always_call_bot.py"),
    ("Always All-In Shove Heuristic", "example_bots/always_shove_bot.py"),
    ("Always Min-Raise Spammer", "example_bots/always_minraise_bot.py"),
    ("Always 100% Pot-Bet Spammer", "example_bots/always_potbet_bot.py"),
]

PROVEN_ALGOS = [
    ("CFR Nash Regret-Matching Bot", "example_bots/cfr_nash_bot.py"),
    ("UAlberta EHS & EHS^2 (Poki/Loki)", "example_bots/ehu_sankey_bot.py"),
    ("DeepStack/Pluribus Geometric/Kelly", "example_bots/kelly_pot_geometric_bot.py"),
    ("Bayesian Combinatorial Range Bot", "example_bots/bayesian_range_bot.py"),
    ("Tartanian/Libratus Hybrid + Blocker", "example_bots/uchicago_slumbot_hybrid.py"),
]

ARCHETYPE_ZOO = [
    ("Starter Passive Bot", "example_bots/starter_bot.py"),
    ("Random Bot", "example_bots/random_bot.py"),
    ("Calling Station Bot", "example_bots/calling_station_bot.py"),
    ("Tight-Aggressive (TAG) Bot", "example_bots/tag_bot.py"),
    ("LAG Maniac Bully Bot", "example_bots/lag_maniac_bot.py"),
    ("Deceptive Trapper Bot", "example_bots/trapper_bot.py"),
    ("CFR / GTO Balanced Bot", "example_bots/cfr_gto_bot.py"),
    ("Rival Adaptive Gear-Bot", "example_bots/rival_adaptive_bot.py"),
]

MIRRORS = [
    ("Self-Play Mirror (Adaptive v4)", "example_bots/adaptive_v4_mirror.py"),
    ("Self-Play Mirror (Adaptive v3)", "example_bots/adaptive_v3_mirror.py"),
]

FULL_20_BOT_SUITE = EXTREME_HEURISTICS + ARCHETYPE_ZOO + PROVEN_ALGOS + MIRRORS


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full20"
    hero = sys.argv[2] if len(sys.argv) > 2 else "bot.py"
    if mode == "proven":
        run_suite(hero, PROVEN_ALGOS + MIRRORS, hands_per_match=300, seeds=(101, 202, 303, 404), title="PROVEN ALGORITHMS & SELF-PLAY MIRRORS")
    elif mode == "heuristics":
        run_suite(hero, EXTREME_HEURISTICS + MIRRORS, hands_per_match=300, seeds=(101, 202, 303, 404), title="EXTREME HEURISTICS & SELF-PLAY MIRRORS")
    else:
        run_suite(hero, FULL_20_BOT_SUITE, hands_per_match=300, seeds=(101, 202, 303), title="FULL 20-BOT CHAMPIONSHIP SUITE")
