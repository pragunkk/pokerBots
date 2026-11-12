# Discard Hold'em — Adaptive Meta-Bot (`v8`)

Flagship bot and tournament testing suite for the **Pokerbots Discard Hold'em Hackathon** (Python 3.11 standard library only, official Two-Leg duplicate match format).

## 1. Project Structure

```text
pokerBots/
├── Starter_Kit/
│   ├── bot.py                # Standalone single-file tournament submission (built via tools/build_bot.py)
│   ├── adaptive_bot.py       # Core AdaptiveBot v8 strategy, street-level profiler & RangeWeight engine
│   ├── fast_eval.py          # Startup O(1) POW5/Bitmask 5/6/7-card evaluator, discard optimizer & exact river solver
│   ├── preflop_table.py      # Precomputed lookup table for all 1,755 canonical 3-card preflop classes
│   ├── engine.py             # Official Two-Leg match runner (Stage 1 = 400 hands, Stage 2 = 1,000 hands)
│   ├── run_tournament.py     # Head-to-head series, Round-Robin, and Swiss tournament runner
│   └── example_bots/         # 28 internal sparring bots (GTO/CFR, Bayesian, heuristics & version mirrors)
├── spar/
│   ├── spar_bot.py           # Complementary ML + Quantile-Bayesian EV sparring bot (2-layer NN + online SGD)
│   ├── train_ml_spar.py      # Pure-Python trainer for the 12-feature postflop neural response model
│   └── ml_weights.json       # Pre-trained weights for the neural response model
├── champions/                # Frozen historical flagship checkpoints (v5.py, v6.py, v6_2.py, v7.py, v8.py)
├── opponentBots/             # Practice bots from other teams (bots_Bhavesh.py, bot_Varshith.py, cfr_nash_bot.py)
├── tools/
│   ├── build_bot.py          # Bundles preflop_table.py + fast_eval.py + adaptive_bot.py -> Starter_Kit/bot.py
│   ├── benchmark_opponent.py # Runs Stage 1 + Stage 2 matches vs any bot and outputs a Markdown forensics report
│   ├── h2h.py                # Fast parallel head-to-head evaluator with mean ± standard error verdict
│   ├── gauntlet.py           # Runs a bot against all internal bots in Starter_Kit/example_bots/
│   ├── fuzz.py               # Stress-tests a bot with random game states to verify 0 crashes/illegal actions
│   └── gen_preflop.py        # Offline generator for the 1,755-class preflop equity table
├── reports/                  # Generated tournament benchmark & street-level forensics reports
└── docs/                     # Official problemStatement.pdf, ruleBook.pdf, and protocol.pdf
```

## 2. Flagship Bot Strategy (`Starter_Kit/bot.py` — `v8`)

Our flagship bot (**Adaptive Meta-Bot v8**) combines teammate `v7`'s soft preflop range-weighting engine with sparring-driven structural upgrades:

1. **Startup $O(1)$ Lookup-Table Hand Evaluator (`fast_eval.py`)**:
   - Precomputes `NON_FLUSH_LUT`, `FLUSH_BIT_LUT`, `N6`, `N7`, `FL`, and `FSUIT` using base-5 rank multisets (`POW5`) and 16-bit suit counters (`SUITADD`) during the free 5s startup window.
   - Evaluates any 5-, 6-, or 7-card hand in $O(1)$ time (`< 1.4ms` max decision latency, `~0.45s` per 400-hand Stage 1 match), enabling high-precision Flop (`520` rollouts) and Turn (`720` rollouts) Monte Carlo simulations alongside a zero-variance 903-hand exact combinatorial river solver.
2. **Soft Sigmoid Preflop Range Weighting (`RangeWeight` & `strength_quantile`)**:
   - Precomputes the exact strength distribution of all $\binom{52}{3} = 22,100$ starting hands during `init` and maps the opponent's measured preflop frequencies (`btn_open_rate`, `pfr`, `three_bet_vs_open_rate`, `fold_to_steal`) to smooth double-logistic lower (`lo`) and upper (`hi`, capped-range) weights on their 3-card holding `(opp0, opp1, opp_discard)`.
3. **True 8-Out Straight Draw Filter & Out-of-Position Flop Donk-Bet Guard**:
   - Distinguishes genuine 8-out open-ended and double-gutshot straight draws from 4-out single gutshots in `has_strong_draw`, eliminating loose flop calls with bare 4-out gutshots that previously folded on the turn.
   - Prevents out-of-position flop donk-betting (`self.seat == "big_blind"` and `opp_was_prev_aggressor`) with bluffs and medium 1-pair hands into the Button preflop raiser, while preserving prior-street aggression memory (`opp_raised_us_flop`) on the Turn and River.
4. **Street-Specific Flop Probe & MDF-Balanced Turn/River Defense**:
   - Tracks `fold_to_flop_bet` and `turn_barrel_rate` independently of overall postflop aggression: targets fit-or-fold flop players (`fold_to_flop_bet > 44%`) with `0.35x` pot small-ball flop probes (`68%` frequency) without misclassifying active 3-bettors or turn barrelers (`turn_barrel_rate > 36%`) as passive `NIT_TAG`s.
   - Folds bare One Pair / Overpair (`inter <= 2`) against massive river overbet jams (`to_call >= 60` and $\ge 0.85\times$ pot) while defending standard turn/river barrels at Minimum Defense Frequency (`< 42%` fold rate).

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
python3 Starter_Kit/engine.py Starter_Kit/bot.py spar/spar_bot.py

# Stage 2 Finals Match (1,000 hands = 2 legs of 500 deals, 75s bank/leg, Cap C = 1000)
python3 Starter_Kit/engine.py Starter_Kit/bot.py opponentBots/bot_Varshith.py --stage 2
```

### Run Full Tournament Benchmarks with Forensics Table (`benchmark_opponent.py`)
Runs 5 Stage-1 matches + 2 Stage-2 matches (4,000 hands total) in parallel and outputs match tables + street-by-street forensics:
```bash
python3 tools/benchmark_opponent.py Starter_Kit/bot.py spar/spar_bot.py --output-md reports/benchmark_vs_SparBot.md
python3 tools/benchmark_opponent.py Starter_Kit/bot.py opponentBots/bot_Varshith.py --output-md reports/benchmark_vs_Varshith.md
python3 tools/benchmark_opponent.py Starter_Kit/bot.py opponentBots/bots_Bhavesh.py --output-md reports/benchmark_vs_Bhavesh.md
```

### Run Head-to-Head (`h2h.py`), Gauntlet (`gauntlet.py`), or Multi-Bot Tournaments (`run_tournament.py`)
```bash
# Fast parallel head-to-head over N seeds with statistical significance verdict
python3 tools/h2h.py Starter_Kit/bot.py champions/v7.py 50

# Gauntlet against all internal example bots
python3 tools/gauntlet.py Starter_Kit/bot.py --seeds 4

# Round-Robin or Swiss tournament across multiple bots
python3 Starter_Kit/run_tournament.py --round-robin \
  Starter_Kit/bot.py \
  spar/spar_bot.py \
  opponentBots/bot_Varshith.py \
  opponentBots/bots_Bhavesh.py
```

- **v9**: Lowered 3-bet defense scaling trigger to 9%, added Top-Pair turn MDF shield vs. double-barrelers, OOP turn check-raise value traps (`inter >= 3`), and U-shaped polarized turn/river range weighting.
