# Discard Hold'em — Adaptive Meta-Bot (`v6.2`)

Flagship bot and tournament testing suite for the **Pokerbots Discard Hold'em Hackathon** (Python 3.11 standard library only, official Two-Leg duplicate match format).

## 1. Project Structure

```text
pokerBots/
├── Starter_Kit/
│   ├── bot.py                # Standalone single-file tournament submission (built via tools/build_bot.py)
│   ├── adaptive_bot.py       # Core AdaptiveBot v6.2 strategy & dual-window opponent profiler
│   ├── fast_eval.py          # Sorting-network 5/7-card evaluator, flop discard optimizer & exact river solver
│   ├── preflop_table.py      # Precomputed lookup table for all 1,755 canonical 3-card preflop classes
│   ├── engine.py             # Official Two-Leg match runner (Stage 1 = 400 hands, Stage 2 = 1,000 hands)
│   ├── run_tournament.py     # Head-to-head series, Round-Robin, and Swiss tournament runner
│   └── example_bots/         # 26 internal sparring bots (GTO/CFR, Bayesian, heuristics & prior mirrors)
├── opponentBots/             # Practice bots from other teams (bot_Bhavesh.py, bot_Varshith.py)
├── tools/
│   ├── build_bot.py          # Bundles preflop_table.py + fast_eval.py + adaptive_bot.py -> Starter_Kit/bot.py
│   ├── benchmark_opponent.py # Runs Stage 1 + Stage 2 matches vs any bot and outputs a Markdown forensics report
│   ├── h2h.py                # Fast parallel head-to-head evaluator with mean ± standard error verdict
│   ├── gauntlet.py           # Runs a bot against all 26 bots in Starter_Kit/example_bots/
│   ├── fuzz.py               # Stress-tests a bot with random game states to verify 0 crashes/illegal actions
│   └── gen_preflop.py        # Offline generator for the 1,755-class preflop equity table
├── reports/                  # Generated tournament benchmark & street-level forensics reports
└── docs/                     # Official problemStatement.pdf, ruleBook.pdf, and protocol.pdf
```

## 2. Flagship Bot Strategy (`Starter_Kit/bot.py`)

Our bot (**Adaptive Meta-Bot v6.2**) combines fast exact/Monte Carlo hand evaluation with online opponent profiling:

1. **1,755-Class Preflop Lookup & Anti-3-Bet Mixing**:
   - Uses an $O(1)$ lookup table (`preflop_table.py`) covering all 1,755 canonical 3-card isomorphism classes.
   - Adapts Button opening and Big Blind defense dynamically: auto-steals against tight folders (`fold_to_steal > 62%`), 3-bets wider (`>= 0.59`) in the Big Blind against wide Button openers (`btn_open_rate > 58%`), and mixes 1-chip Button limps (`0.20–0.44`) against aggressive 3-bettors (`three_bet_vs_open_rate > 16%`) to avoid bleeding chips to light 3-bets.
2. **Simultaneous Flop Discard Optimizer**:
   - Evaluates all 3 discard options using time-scaled Monte Carlo rollouts (`220` down to `40` rollouts based on remaining time per hand) weighted by opponent VPIP, nut-potential implied odds, and deterministic made-hand/kicker tie-breaking.
3. **5-Tier Board-Relative Hand Interaction & Minimum Defense Frequency (MDF)**:
   - Classifies hands relative to board texture (`0` = High Card/Board Pair, `1` = Underpair/Middle Pair, `2` = Top Pair/Overpair, `3` = Two Pair/Trips, `4` = Monster Straight/Flush/Full House) alongside `board_danger_level`.
   - Enforces a **Flop Unimproved Air Guard** (folding `inter == 0` without a draw against $\ge 42\%$ pot bets) while maintaining **Minimum Defense Frequency (MDF)** on the Turn and River against aggressive probe/barrel opponents (`postflop_agg_per_hand > 0.58` or `raise_my_bet_rate > 0.20`) so opponents cannot profitably auto-bluff `0.60x` pot.
4. **Zero-Variance Exact River Solver & Value-Raise Safety Guards**:
   - Enumerates all $\binom{43}{2} = 903$ possible opponent 2-card river holdings conditioned on the exposed flop discard (`opp_discard`).
   - Prevents overplaying vulnerable hands at deep stacks by never value-raising river bets with bare Two Pair or non-nut Straights on 3-flush/paired boards.
5. **Crash-Proof Protocol Loop**:
   - Wraps every action and discard decision in `legalize()` and fallback handlers so the bot never forfeits or emits an illegal action.

## 3. How to Build, Run & Test

Run all commands from the repository root:

### Build & Fuzz-Test `Starter_Kit/bot.py`
Whenever you edit `preflop_table.py`, `fast_eval.py`, or `adaptive_bot.py`, rebuild the single-file submission and verify protocol safety:
```bash
python3 tools/build_bot.py
python3 tools/fuzz.py Starter_Kit/bot.py 2000
```

### Play a Single Match (`engine.py`)
```bash
# Stage 1 Match (400 hands = 2 legs of 200 deals, 30s bank/leg, Cap C = 400)
python3 Starter_Kit/engine.py Starter_Kit/bot.py opponentBots/bot_Varshith.py

# Stage 2 Finals Match (1,000 hands = 2 legs of 500 deals, 75s bank/leg, Cap C = 1000)
python3 Starter_Kit/engine.py Starter_Kit/bot.py opponentBots/bot_Bhavesh.py --stage 2
```

### Run Full Tournament Benchmarks with Forensics Table (`benchmark_opponent.py`)
Runs 5 Stage-1 matches + 2 Stage-2 matches (4,000 hands total) in parallel and outputs match tables + street-by-street forensics:
```bash
python3 tools/benchmark_opponent.py Starter_Kit/bot.py opponentBots/bot_Varshith.py --output-md reports/benchmark_vs_Varshith.md
python3 tools/benchmark_opponent.py Starter_Kit/bot.py opponentBots/bot_Bhavesh.py --output-md reports/benchmark_vs_Bhavesh.md
```

### Run Head-to-Head (`h2h.py`), Gauntlet (`gauntlet.py`), or Multi-Bot Tournaments (`run_tournament.py`)
```bash
# Fast parallel head-to-head over N seeds with statistical significance verdict
python3 tools/h2h.py Starter_Kit/bot.py Starter_Kit/example_bots/mirror_v6.py 50

# Gauntlet against all 26 internal example bots
python3 tools/gauntlet.py Starter_Kit/bot.py --seeds 4

# Round-Robin or Swiss tournament across multiple bots
python3 Starter_Kit/run_tournament.py --round-robin \
  Starter_Kit/bot.py \
  opponentBots/bot_Varshith.py \
  opponentBots/bot_Bhavesh.py \
  Starter_Kit/example_bots/cfr_nash_bot.py
```
