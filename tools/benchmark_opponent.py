#!/usr/bin/env python3
"""
Comprehensive Tournament Benchmark & Forensics Runner for Discard Hold'em.
Tests our flagship Starter_Kit/bot.py against any opponent bot (e.g. opponentBots/bot_Varshith.py)
under official Two-Leg engine rules and outputs clean tabular summaries & deep hand forensics.
"""

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR) if os.path.basename(BASE_DIR) in ("tools", "Starter_Kit") else BASE_DIR
ENGINE = os.path.join(REPO_ROOT, "Starter_Kit", "engine.py")

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


def run_single_match(bot_a, bot_b, stage=1, hands=None, seed=None, log_dir="", python_exe=sys.executable):
    st = STAGES[stage]
    match_hands = hands or st["hands"]
    cap = st["cap"]
    seed_str = str(seed if seed is not None else 1001)

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
    res = subprocess.run(cmd, cwd=os.path.dirname(ENGINE), capture_output=True, text=True)
    out = res.stdout

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
            "log_dir": log_dir,
        }

    net_a = int(m_a.group(1))
    time_a = float(m_a.group(2))
    notices_a = int(m_a.group(3))
    net_b = int(m_b.group(1))
    time_b = float(m_b.group(2))
    notices_b = int(m_b.group(3))
    played = int(m_played.group(1)) if m_played else match_hands

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
        "log_dir": log_dir,
    }


def parse_match_logs(log_dirs):
    total_hands = 0
    a_net = 0
    b_folds, a_folds = 0, 0
    showdowns, a_sd_wins, b_sd_wins, splits = 0, 0, 0, 0

    b_pre_folds, b_flop_folds, b_turn_folds, b_river_folds = 0, 0, 0, 0
    a_pre_folds, a_flop_folds, a_turn_folds, a_river_folds = 0, 0, 0, 0

    a_btn_hands, a_btn_opens, b_folded_to_a_open = 0, 0, 0
    b_btn_hands, b_btn_opens, a_folded_to_b_open = 0, 0, 0

    b_3bets_pre, a_3bets_pre = 0, 0
    big_pots_a = []
    big_pots_b = []

    for ldir in log_dirs:
        mpath = os.path.join(ldir, "match.log")
        if not os.path.isfile(mpath):
            continue
        with open(mpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("L") or "|" not in line:
                    continue
                total_hands += 1
                m_net = re.search(r"A ([+-]\d+)", line)
                net_a = int(m_net.group(1)) if m_net else 0
                a_net += net_a

                m_pre = re.search(r"preflop:\s+(.*?)(?:\s+flop:|\s+\|)", line)
                pre_seg = m_pre.group(1) if m_pre else ""

                is_btn_a = "button A" in line
                if is_btn_a:
                    a_btn_hands += 1
                    if "Ar" in pre_seg:
                        a_btn_opens += 1
                        if "Bf" in pre_seg and "Br" not in pre_seg:
                            b_folded_to_a_open += 1
                        if "Br" in pre_seg:
                            b_3bets_pre += 1
                else:
                    b_btn_hands += 1
                    if "Br" in pre_seg:
                        b_btn_opens += 1
                        if "Af" in pre_seg and "Ar" not in pre_seg:
                            a_folded_to_b_open += 1
                        if "Ar" in pre_seg:
                            a_3bets_pre += 1

                if "fold" in line:
                    if net_a > 0:
                        b_folds += 1
                        if " flop:" not in line: b_pre_folds += 1
                        elif " turn:" not in line: b_flop_folds += 1
                        elif " river:" not in line: b_turn_folds += 1
                        else: b_river_folds += 1
                    else:
                        a_folds += 1
                        if " flop:" not in line: a_pre_folds += 1
                        elif " turn:" not in line: a_flop_folds += 1
                        elif " river:" not in line: a_turn_folds += 1
                        else: a_river_folds += 1
                elif "showdown" in line:
                    showdowns += 1
                    if net_a > 0: a_sd_wins += 1
                    elif net_a < 0: b_sd_wins += 1
                    else: splits += 1

                if net_a <= -40:
                    big_pots_b.append((net_a, line))
                elif net_a >= 40:
                    big_pots_a.append((net_a, line))

    return {
        "total_hands": total_hands,
        "a_net": a_net,
        "b_folds": b_folds,
        "b_pre_folds": b_pre_folds,
        "b_flop_folds": b_flop_folds,
        "b_turn_folds": b_turn_folds,
        "b_river_folds": b_river_folds,
        "a_folds": a_folds,
        "a_pre_folds": a_pre_folds,
        "a_flop_folds": a_flop_folds,
        "a_turn_folds": a_turn_folds,
        "a_river_folds": a_river_folds,
        "showdowns": showdowns,
        "a_sd_wins": a_sd_wins,
        "b_sd_wins": b_sd_wins,
        "splits": splits,
        "a_btn_hands": a_btn_hands,
        "a_btn_opens": a_btn_opens,
        "b_folded_to_a_open": b_folded_to_a_open,
        "b_3bets_pre": b_3bets_pre,
        "b_btn_hands": b_btn_hands,
        "b_btn_opens": b_btn_opens,
        "a_folded_to_b_open": a_folded_to_b_open,
        "a_3bets_pre": a_3bets_pre,
        "big_pots_a": big_pots_a,
        "big_pots_b": big_pots_b,
    }


def run_benchmark_tournament(bot_a, bot_b, s1_seeds=(1001, 1102, 1203, 1304, 1405), s2_seeds=(777, 888), parallel=4, output_md=""):
    name_a = short_name(bot_a)
    name_b = short_name(bot_b)
    log_base = os.path.join(REPO_ROOT, "Starter_Kit", "logs", f"bench_{name_a}_vs_{name_b}")
    os.makedirs(log_base, exist_ok=True)

    print(f"\n================================================================================================")
    print(f"  STARTING OFFICIAL TOURNAMENT BENCHMARK: {name_a} (A) vs {name_b} (B)")
    print(f"  Stage 1: {len(s1_seeds)} Matches x 400 Hands (2 Legs of 200 Deals) | Cap C = 400")
    print(f"  Stage 2: {len(s2_seeds)} Matches x 1,000 Hands (2 Legs of 500 Deals) | Cap C = 1000")
    print(f"================================================================================================")

    s1_tasks = []
    s1_log_dirs = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for idx, s in enumerate(s1_seeds):
            ldir = os.path.join(log_base, f"s1_match_{idx + 1}_seed_{s}")
            s1_log_dirs.append(ldir)
            s1_tasks.append((idx + 1, s, ex.submit(run_single_match, bot_a, bot_b, 1, 400, s, ldir)))

    s1_results = [fut.result() for _, _, fut in s1_tasks]

    s2_tasks = []
    s2_log_dirs = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        for idx, s in enumerate(s2_seeds):
            ldir = os.path.join(log_base, f"s2_match_{idx + 1}_seed_{s}")
            s2_log_dirs.append(ldir)
            s2_tasks.append((idx + 1, s, ex.submit(run_single_match, bot_a, bot_b, 2, 1000, s, ldir)))

    s2_results = [fut.result() for _, _, fut in s2_tasks]

    combined_stats = parse_match_logs(s1_log_dirs + s2_log_dirs)

    lines = []
    lines.append(f"# Official Tournament Benchmark Report: `{name_a}` vs `{name_b}`\n")
    lines.append(f"**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).")
    lines.append(f"- **Bot A (Hero)**: `{name_a}` (`{os.path.abspath(bot_a)}`)")
    lines.append(f"- **Bot B (Opponent)**: `{name_b}` (`{os.path.abspath(bot_b)}`)\n")

    lines.append("## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)")
    lines.append("| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
    for idx, r in enumerate(s1_results, 1):
        win_str = f"**A ({name_a})**" if r["pts_a"] == 1.0 else (f"B ({name_b})" if r["pts_b"] == 1.0 else "DRAW")
        if "FORFEIT" in r["status"]: win_str += "*"
        t_str = f"{r['time_a']:5.2f}s / {r['time_b']:5.2f}s"
        n_str = f"{r['notices_a']} / {r['notices_b']}"
        lines.append(f"| Match #{idx} | `{r['seed']}` | {win_str} | `{r['net_a']:+d}` | `{r['capped_a']:+d}` | {t_str} | {n_str} |")

    s1_w = sum(1 for r in s1_results if r["pts_a"] == 1.0)
    s1_d = sum(1 for r in s1_results if r["pts_a"] == 0.5)
    s1_l = sum(1 for r in s1_results if r["pts_a"] == 0.0)
    s1_pts = sum(r["pts_a"] for r in s1_results)
    s1_raw = sum(r["net_a"] for r in s1_results)
    s1_cap = sum(r["capped_a"] for r in s1_results)
    s1_hands = sum(r["hands"] for r in s1_results)
    if s1_hands > 0:
        lines.append(f"| **Stage 1 Total** | **{len(s1_results)} Matches** | **{s1_pts:.1f} / {len(s1_results)} ({s1_w}W-{s1_d}D-{s1_l}L)** | **`{s1_raw:+d}` ({s1_raw/s1_hands:+.3f}/h)** | **Score_cap = `{s1_cap/s1_hands:+.4f}`** | **—** | **0 / 0** |\n")

    if s2_results:
        lines.append("## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)")
        lines.append("| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |")
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")
        for idx, r in enumerate(s2_results, 1):
            win_str = f"**A ({name_a})**" if r["pts_a"] == 1.0 else (f"B ({name_b})" if r["pts_b"] == 1.0 else "DRAW")
            if "FORFEIT" in r["status"]: win_str += "*"
            t_str = f"{r['time_a']:5.2f}s / {r['time_b']:5.2f}s"
            n_str = f"{r['notices_a']} / {r['notices_b']}"
            lines.append(f"| Match #{idx} | `{r['seed']}` | {win_str} | `{r['net_a']:+d}` | `{r['capped_a']:+d}` | {t_str} | {n_str} |")

        s2_w = sum(1 for r in s2_results if r["pts_a"] == 1.0)
        s2_d = sum(1 for r in s2_results if r["pts_a"] == 0.5)
        s2_l = sum(1 for r in s2_results if r["pts_a"] == 0.0)
        s2_pts = sum(r["pts_a"] for r in s2_results)
        s2_raw = sum(r["net_a"] for r in s2_results)
        s2_cap = sum(r["capped_a"] for r in s2_results)
        s2_hands = sum(r["hands"] for r in s2_results)
        lines.append(f"| **Stage 2 Total** | **{len(s2_results)} Matches** | **{s2_pts:.1f} / {len(s2_results)} ({s2_w}W-{s2_d}D-{s2_l}L)** | **`{s2_raw:+d}` ({s2_raw/s2_hands:+.3f}/h)** | **Score_cap = `{s2_cap/s2_hands:+.4f}`** | **—** | **0 / 0** |\n")
    else:
        s2_pts = s2_raw = s2_cap = s2_hands = 0

    grand_pts = s1_pts + s2_pts
    grand_matches = max(1, len(s1_results) + len(s2_results))
    grand_raw = s1_raw + s2_raw
    grand_hands = max(1, s1_hands + s2_hands)
    grand_cap = s1_cap + s2_cap
    lines.append("## 3. Overall Tournament Aggregate Summary")
    lines.append(f"- **Total Hands Played**: `{grand_hands}` hands across `{grand_matches}` matches")
    lines.append(f"- **Match Points**: **`{grand_pts:.1f} / {grand_matches}` ({100*grand_pts/grand_matches:.1f}% Win Rate)**")
    lines.append(f"- **Official Score_cap**: **`{grand_cap/grand_hands:+.4f}` / hand**")
    lines.append(f"- **Total Raw Net Chips**: **`{grand_raw:+d}` chips (`{grand_raw/grand_hands:+.3f}` chips/hand)**\n")

    lines.append("## 4. Deep Hand-Level & Street-by-Street Forensics")
    lines.append(f"| Metric Category | Hero: `{name_a}` (A) | Opponent: `{name_b}` (B) | Strategic Insight |")
    lines.append("| :--- | :---: | :---: | :--- |")

    tot_h = max(1, combined_stats["total_hands"])
    a_btn_open_r = combined_stats["a_btn_opens"] / max(1, combined_stats["a_btn_hands"])
    b_btn_open_r = combined_stats["b_btn_opens"] / max(1, combined_stats["b_btn_hands"])
    b_fold_to_a = combined_stats["b_folded_to_a_open"] / max(1, combined_stats["a_btn_opens"])
    a_fold_to_b = combined_stats["a_folded_to_b_open"] / max(1, combined_stats["b_btn_opens"])

    lines.append(f"| **Button Open Frequency** | `{a_btn_open_r*100:.1f}%` ({combined_stats['a_btn_opens']}/{combined_stats['a_btn_hands']}) | `{b_btn_open_r*100:.1f}%` ({combined_stats['b_btn_opens']}/{combined_stats['b_btn_hands']}) | Preflop steal aggression |")
    lines.append(f"| **Fold to Preflop Open (BB)** | `{a_fold_to_b*100:.1f}%` ({combined_stats['a_folded_to_b_open']}/{combined_stats['b_btn_opens']}) | `{b_fold_to_a*100:.1f}%` ({combined_stats['b_folded_to_a_open']}/{combined_stats['a_btn_opens']}) | Big Blind defense tightness |")
    lines.append(f"| **Preflop 3-Bet Frequency** | `{combined_stats['a_3bets_pre']}` times | `{combined_stats['b_3bets_pre']}` times | Preflop re-raise pressure |")

    b_tot_folds = combined_stats["b_folds"]
    a_tot_folds = combined_stats["a_folds"]
    lines.append(f"| **Total Folds Across Hands** | `{a_tot_folds}` ({a_tot_folds/tot_h*100:.1f}%) | `{b_tot_folds}` ({b_tot_folds/tot_h*100:.1f}%) | Overall fold frequency |")
    lines.append(f"| **Preflop Folds** | `{combined_stats['a_pre_folds']}` ({combined_stats['a_pre_folds']/tot_h*100:.1f}%) | `{combined_stats['b_pre_folds']}` ({combined_stats['b_pre_folds']/tot_h*100:.1f}%) | Hand surrendered preflop |")
    lines.append(f"| **Flop Folds** | `{combined_stats['a_flop_folds']}` | `{combined_stats['b_flop_folds']}` | C-bet / probe fold frequency |")
    lines.append(f"| **Turn Folds** | `{combined_stats['a_turn_folds']}` | `{combined_stats['b_turn_folds']}` | Turn barrel fold frequency |")
    lines.append(f"| **River Folds** | `{combined_stats['a_river_folds']}` | `{combined_stats['b_river_folds']}` | River fold frequency |")

    sd = combined_stats["showdowns"]
    lines.append(f"| **Showdown Reach Frequency** | `{sd}` hands ({sd/tot_h*100:.1f}%) | `{sd}` hands ({sd/tot_h*100:.1f}%) | Hands reaching final showdown |")
    lines.append(f"| **Showdowns Won** | `{combined_stats['a_sd_wins']}` ({combined_stats['a_sd_wins']/max(1, sd)*100:.1f}%) | `{combined_stats['b_sd_wins']}` ({combined_stats['b_sd_wins']/max(1, sd)*100:.1f}%) | Showdown equity conversion |")
    lines.append(f"| **Big Pots Won (>= 40 chips)** | **`{len(combined_stats['big_pots_a'])}`** pots | **`{len(combined_stats['big_pots_b'])}`** pots | High-leverage stack-off edge |\n")

    report_text = "\n".join(lines)
    print(report_text)

    if output_md:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\nReport written cleanly to: {output_md}")

    return {
        "s1_results": s1_results,
        "s2_results": s2_results,
        "combined_stats": combined_stats,
        "report_text": report_text,
    }


def main():
    p = argparse.ArgumentParser(description="Official Tournament Benchmark & Forensics Runner")
    p.add_argument("bot_a", help="Path to bot A (Hero)")
    p.add_argument("bot_b", help="Path to bot B (Opponent)")
    p.add_argument("--s1-matches", type=int, default=5, help="Number of Stage 1 matches (default: 5)")
    p.add_argument("--s2-matches", type=int, default=2, help="Number of Stage 2 matches (default: 2)")
    p.add_argument("--output-md", default="reports/benchmark_vs_Varshith.md", help="Output Markdown report path")
    p.add_argument("--parallel", type=int, default=4, help="Max parallel matches")
    args = p.parse_args()

    s1_seeds = [1001 + i * 101 for i in range(args.s1_matches)]
    s2_seeds = [777 + i * 111 for i in range(args.s2_matches)]

    if args.output_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)

    run_benchmark_tournament(args.bot_a, args.bot_b, s1_seeds, s2_seeds, args.parallel, args.output_md)


if __name__ == "__main__":
    main()
