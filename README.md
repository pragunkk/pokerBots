# Pokerbots Hackathon: Discard Hold'em — Adaptive Meta-Bot v5

Championship submission and multi-archetype testing framework for **Discard Hold'em** (Python 3.11, protocol v1.0).

## 1. Repository Layout

| Path | Description |
| :--- | :--- |
| `Starter_Kit/bot.py` | **Standalone Championship Submission Bot (Adaptive Meta-Bot v5)** — self-contained single-file bundle (`96.9 KB`, standard library only, zero runtime file I/O, `< 1.5s` per 1,000-hand match). |
| `Starter_Kit/adaptive_bot.py` | Modular source for **Adaptive Meta-Bot v5** (`OpponentProfile` dual-window tracker, multi-gear selector, and match-point variance controller). |
| `Starter_Kit/fast_eval.py` | Sorting-network 5/7-card hand evaluator, `hole_interaction_level` classifier, `board_danger_level` analyzer, rollout-based flop discard optimizer, and Bayesian turn/river equity engine. |
| `Starter_Kit/preflop_table.py` | Embedded lookup table for all **1,755 canonical 3-card preflop isomorphism classes** in Discard Hold'em. |
| `Starter_Kit/tournament_sim.py` | Parallel multi-seed tournament simulator across the **20-Bot Championship Suite** (5 Extreme Heuristics, 8 Archetypes, 5 Proven Computer-Poker Algorithms, and 2 Self-Play Evolutionary Mirrors). |
| `Starter_Kit/engine.py` | Official Discard Hold'em local match runner. |
| `STRATEGY_AND_ROADMAP.md` | Full mathematical analysis, exploit forensics, and evolutionary roadmap (`v1` $\to$ `v5`). |

## 2. Core Architecture & Novel Exploits (Adaptive Bot v5)

1. **`O(1)` 1,755 Canonical 3-Card Preflop Lookup (`preflop_table.py`)**:
   - Maps any 3-card hand to its exact suit-isomorphic canonical tuple `(r0, r1, r2, suit_pattern)` and looks up precomputed post-flop-discard equity in `0.001 ms` with zero file I/O.
2. **Deterministic Flop Made-Hand & Nut-Potential ($P_{\text{pot}}$) Discard Engine (`fast_eval.py`)**:
   - Evaluates all 3 discard choices via VPIP-conditioned Monte Carlo rollouts augmented by a `3.5%` implied-odds bonus for nut-making outs (`cat >= 3`) and a deterministic current flop made-hand + kicker tie-breaker so rollout variance never breaks a higher pocket pair (e.g., keeping `6d 6h` over `4d 4c` on `4c 9h Jd`) or discards an Ace/King kicker.
3. **Bayesian Discard Reader & Deceptive-Discard Immunity**:
   - Filters opponent Turn/River holdings by whether `opp_discard` was rationally discarded on the flop (`weight *= 0.08` for irrational keeps) and monitors `high_card_discards / total_discards`. If an opponent frequently discards face cards (`> 22%`) as a phantom-strength bluff, the bot automatically disables the high-card discard equity penalty.
4. **Board-Relative Hand Interaction & Wet-Board Overbet Defense (`hole_interaction_level` & `board_danger_level`)**:
   - Distinguishes High Card / Board-Only Pairs (`inter = 0`), Underpairs / Weak Pairs (`inter = 1`), Top Pair / Overpair (`inter = 2`, even when the board has a secondary pair), Two Pair / 1-Card Trips (`inter = 3`), and Pocket Sets / Straights / Flushes / Full Houses (`inter = 4`).
   - Prevents calling Turn barrels (`>= 28%` pot) with unimproved High Card (`inter = 0`) when holding no 4-flush/4-straight draw (`has_strong_draw`), and prevents 3-betting the River with 1-card Trips or bare Two Pair into Sets/Straights/Full Houses.
5. **Extreme Heuristic & Boundary Crushers**:
   - **Auto-Steal vs. Extreme Folders (`fold_to_steal > 0.62` or `fold_to_bet > 0.68`)**: Opens `100%` of Buttons to `4` chips (min-raise) and fires `100%` min-probe continuation bets, achieving the theoretical maximum `+1.500` chips/hand (`+450` in 300 hands) against `Always Fold`.
   - **Anti-Shove-Spammer Shield (`preflop_shove_rate > 0.25`)**: Never open-raises medium hands (`< 0.57`) on the Button only to fold to an automatic `200`-chip shove (saving `4–5` chips per hand), and calls all-in shoves with the exact dominating range (`strength >= 0.57`, `+11.27` chips/hand) without `PROTECT_WIN` over-folding.
6. **Asymmetric Capped-Range River Steal & Bluff-Catching (`META_HUNTER` v5 Exploit)**:
   - When the Turn checks through (`prev_checked_through`), balanced/GTO/v4-mirror opponents are capped at `inter <= 1`. Our bot sizes river bluffs (`inter == 0`) at `int(pot * 0.38) + 1` (`> 26%` of new total pot) to force threshold bots to fold `inter <= 1`, while defending our own `inter == 1` hands up to `33%` of total pot (`0.49x` pot) to snap off opponent `0.35x–0.42x` river bluffs.

## 3. Benchmark Results

### 3.1 Adaptive Bot v5 vs. Proven Computer-Poker Algorithms & Self-Play Mirrors (`300` Hands $\times$ 4 Seeds)

| Opponent Algorithm | W–D–L | Seed Breakdown (`101, 202, 303, 404`) | Total Net Chips | Avg / Hand | Notices |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `cfr_nash_bot.py` (CFR Regret-Matching + MDF) | **4–0–0** | `+731, +994, +870, +1563` | **`+4,158`** | **`+3.465`** | 0 |
| `ehu_sankey_bot.py` (UAlberta Poki/Loki $EHS$ & $EHS^2$) | **4–0–0** | `+934, +73, +1269, +831` | **`+3,107`** | **`+2.589`** | 0 |
| `kelly_pot_geometric_bot.py` (DeepStack/Pluribus Geometric/Kelly) | **4–0–0** | `+671, +542, +846, +1212` | **`+3,271`** | **`+2.726`** | 0 |
| `bayesian_range_bot.py` (Combinatorial Range Particle Filter) | **3–0–1** | `+18, -253, +802, +610` | **`+1,177`** | **`+0.981`** | 0 |
| `uchicago_slumbot_hybrid.py` (Pseudoharmonic + Blocker Bluff) | **4–0–0** | `+371, +62, +713, +965` | **`+2,111`** | **`+1.759`** | 0 |
| `adaptive_v4_mirror.py` (Self-Play Mirror: Adaptive Bot v4) | **4–0–0** | `+346, +485, +50, +441` | **`+1,322`** | **`+1.102`** | 0 |
| `adaptive_v3_mirror.py` (Self-Play Mirror: Adaptive Bot v3) | **4–0–0** | `+217, +538, +394, +326` | **`+1,475`** | **`+1.229`** | 0 |
| **Overall (`8,400` Hands)** | **27–0–1 (`96.4%`)** | — | **`+16,621`** | **`+1.979`** | **0** |

### 3.2 Adaptive Bot v5 vs. Extreme Heuristics (`300` Hands $\times$ 4 Seeds)

| Heuristic Opponent | W–D–L | Seed Breakdown (`101, 202, 303, 404`) | Total Net Chips | Avg / Hand | Notices |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `always_fold_bot.py` (Always Folds to Any Bet) | **4–0–0** | `+450, +450, +450, +450` (100% Max) | **`+1,800`** | **`+1.500`** | 0 |
| `always_call_bot.py` (Calls Every Bet to Showdown) | **4–0–0** | `+3474, +3415, +3681, +3461` | **`+14,031`** | **`+11.693`** | 0 |
| `always_shove_bot.py` (Shoves 200 Chips Every Action) | **4–0–0** | `+1834, +4425, +4825, +2440` | **`+13,524`** | **`+11.270`** | 0 |
| `always_minraise_bot.py` (Spams Min-Raise Every Action) | **4–0–0** | `+4289, +6068, +6212, +5498` | **`+22,067`** | **`+18.389`** | 0 |
| `always_potbet_bot.py` (Raises 100% Pot Every Action) | **4–0–0** | `+2629, +6799, +6102, +4458` | **`+19,988`** | **`+16.657`** | 0 |
| **Overall (`6,000` Hands)** | **20–0–0 (`100.0%`)** | — | **`+71,410`** | **`+11.902`** | **0** |

## 4. Quick Start & Running Benchmarks

```bash
cd Starter_Kit
# Run a single 400-hand Stage 1 match or 1000-hand Stage 2 match
python3 engine.py bot.py example_bots/cfr_nash_bot.py --stage 1
python3 engine.py bot.py example_bots/adaptive_v4_mirror.py --stage 2

# Run benchmark suites
python3 tournament_sim.py proven bot.py
python3 tournament_sim.py heuristics bot.py
python3 tournament_sim.py full20 bot.py
```
