# Pokerbots Hackathon: Discard Hold'em — Adaptive Meta-Bot v6

Championship submission and 24-bot testing framework for **Discard Hold'em** (Python 3.11, protocol v1.0).

## 1. Repository Layout

| Path | Description |
| :--- | :--- |
| `Starter_Kit/bot.py` | **Standalone Championship Submission Bot (Adaptive Meta-Bot v6)** — self-contained single-file bundle (`99.3 KB`, standard library only, zero runtime file I/O, `< 1.8s` per 1,000-hand match). |
| `Starter_Kit/adaptive_bot.py` | Modular source for **Adaptive Meta-Bot v6** (`OpponentProfile` dual-window tracker, multi-gear selector, sub-threshold value optimizer, triple-barrel river defense, and match-point variance controller). |
| `Starter_Kit/fast_eval.py` | Sorting-network 5/7-card hand evaluator, **Full 903-Hand Exact Combinatorial River Solver (`exact_river_equity`)**, **Pre-Evaluated Turn Rollout Engine (`145` rollouts)**, `hole_interaction_level`, `board_danger_level`, and `has_strong_draw`. |
| `Starter_Kit/preflop_table.py` | Embedded lookup table for all **1,755 canonical 3-card preflop isomorphism classes** in Discard Hold'em. |
| `Starter_Kit/tournament_sim.py` | Parallel multi-seed tournament simulator across the **24-Bot Championship Suite** (5 Extreme Heuristics, 8 Archetypes, 5 Proven Computer-Poker Algorithms, 3 Radical Architectures, and 3 Self-Play Evolutionary Mirrors `v3`, `v4`, `v5`). |
| `Starter_Kit/engine.py` | Official Discard Hold'em local match runner. |
| `STRATEGY_AND_ROADMAP.md` | Full mathematical analysis, exploit forensics, radical architectures, and evolutionary roadmap (`v1` $\to$ `v6`). |

## 2. Radical Bot Architectures Explored (`Starter_Kit/example_bots/`)

1. **`mcts_ismcts_bot.py` — Information-Set Monte Carlo Tree Search (IS-MCTS) / Direct EV Tree Planner**:
   - Replaces static threshold rules with explicit Expected-Value maximization over discrete action trees `A = {fold, check/call, raise_0.38x, raise_0.58x, raise_0.85x}` using an online logistic fold-probability model $P(\text{fold} \mid \text{bet\_frac}, \text{street})$ and equity degradation when called.
2. **`contextual_bandit_meta_bot.py` — Online Multiplicative-Weights / Hedge Expert Portfolio Meta-Bot**:
   - Maintains 5 specialist expert policies (`GTO_MDF`, `StationValue`, `NitSteal`, `ManiacTrap`, `SmallBallMeta`), evaluates counterfactual hand rewards after every hand, and updates portfolio weights via exponential multiplicative weights $w_i \leftarrow w_i \exp(\eta \cdot r_i)$.
3. **`exact_bayesian_ev_bot.py` — Exact Zero-Variance Combinatorial River Solver & Probe-Tell Bot**:
   - Replaces noisy River Monte Carlo sampling with exact combinatorial enumeration over the remaining $43$ cards ($\binom{43}{2} = 903$ opponent river holdings) using pre-evaluated single-card flop-discard scores (`alt_flop_score`), plus 150 pre-evaluated Turn rollouts.

## 3. Adaptive Meta-Bot v6 Breakthroughs (`Starter_Kit/bot.py`)

1. **Full 903-Hand Exact Combinatorial River Solver (`exact_river_equity`)**:
   - Precomputes `my_score` once and `alt_flop_score[c] = eval5_idx(c, opp_discard, b0, b1, b2)` in $43$ calls, then evaluates all $\binom{43}{2} = 903$ possible opponent 2-card holdings on the River in `< 1.5 ms` with **0.0% Monte Carlo sampling variance**.
2. **Pre-Evaluated Turn Rollout Accelerator (`145` rollouts)**:
   - Precomputes single-card flop-discard scores once per Turn decision, cutting per-rollout cost by `3x` and increasing Turn rollout precision by `45%`.
3. **Sub-Threshold Value Sizing Optimization (`0.62x` pot)**:
   - Sizes strong Top Pair / Overpair (`inter >= 2, eq_adj >= 0.68`) value bets at `0.62x` pot—extracting `+20%` more chips per street than `0.52x` while staying safely underneath the `0.72x` river overbet-fold threshold of rival bots.
4. **Triple-Barrel One-Pair River Defense (`opp_bet_flop and opp_bet_turn`)**:
   - Recognizes that non-Maniac algorithms (`river_barrel_rate < 0.48`) never triple-barrel Flop + Turn + River (`>= 48%` pot) as a bluff, folding dominated bare One Pair (`inter <= 2, eq_adj < 0.58`) on the River.
5. **Extreme Heuristic & Capped-Range Exploits**:
   - Retains the `100%` Auto-Steal vs. Extreme Folders (`+450/450` vs. `Always Fold`), Anti-Shove-Spammer Shield (`+13,524` vs. `Always Shove`), Turn Unimproved Air Guard (`has_strong_draw`), and Asymmetric Checked-Through River Thresholds.

## 4. Benchmark Results: Adaptive Bot v6 vs. Previous Generations (`v5`, `v4`, `v3`) & Radical Architectures

Tested across `300` duplicate-deal hands/match $\times$ 4 seeds (`1,200` hands per opponent, `7,200` hands total):

| Opponent Architecture / Mirror | W–D–L | Seed Breakdown (`101, 202, 303, 404`) | Total Net Chips | Avg / Hand | Notices |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `adaptive_v5_mirror.py` (Previous Gen: Adaptive Bot v5) | **3–0–1** | `+93, -280, +270, +843` | **`+926`** | **`+0.772`** | 0 |
| `adaptive_v4_mirror.py` (Previous Gen: Adaptive Bot v4) | **4–0–0** | `+127, +484, +318, +221` | **`+1,150`** | **`+0.958`** | 0 |
| `adaptive_v3_mirror.py` (Previous Gen: Adaptive Bot v3) | **3–0–1** | `+190, -276, +517, +207` | **`+638`** | **`+0.532`** | 0 |
| `mcts_ismcts_bot.py` (Radical 1: IS-MCTS Direct EV Tree) | **2–0–2** | `-366, -449, +561, +409` | **`+155`** | **`+0.129`** | 0 |
| `contextual_bandit_meta_bot.py` (Radical 2: Hedge Portfolio) | **3–0–1** | `-295, +253, +398, +1050` | **`+1,406`** | **`+1.172`** | 0 |
| `exact_bayesian_ev_bot.py` (Radical 3: Exact River Solver) | **4–0–0** | `+75, +344, +137, +146` | **`+702`** | **`+0.585`** | 0 |
| **Overall (`7,200` Hands)** | **19–0–5 (`79.2%`)** | — | **`+4,977`** | **`+0.691`** | **0** |

## 5. Quick Start

```bash
cd Starter_Kit
# Run a 400-hand Stage 1 or 1000-hand Stage 2 match
python3 engine.py bot.py example_bots/adaptive_v5_mirror.py --stage 1
python3 engine.py bot.py example_bots/exact_bayesian_ev_bot.py --stage 2

# Run parallel multi-seed benchmark suites
python3 tournament_sim.py proven bot.py
python3 tournament_sim.py heuristics bot.py
python3 tournament_sim.py full20 bot.py
```
