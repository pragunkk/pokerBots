# Discard Hold'em: Adaptive Bot (v10)

Our submission for the **Pokerbots Discard Hold'em** competition. Python 3.11, standard library only, two-leg duplicate match format.

**Submitted file:** `Starter_Kit/bot.py` (v10, SHA-256 starts `afb6fd339b03ef77`). It is generated from three source files by `tools/build_bot.py`. Never edit it by hand.

## Project structure

```text
pokerBots/
├── Starter_Kit/
│   ├── bot.py              # Single-file submission (generated, do not edit)
│   ├── adaptive_bot.py     # Strategy: preflop, postflop, discard, opponent profiler
│   ├── fast_eval.py        # Lookup-table 5/6/7-card evaluator, equity and range functions
│   ├── preflop_table.py    # Strength table for all 1,755 canonical 3-card hands
│   ├── engine.py           # Official two-leg match runner (from the organisers)
│   ├── run_tournament.py   # Round-robin / Swiss runner for local testing
│   └── example_bots/       # Our own sparring bots, version mirrors, chaos and probe bots
├── spar/                   # Our ML sparring bot used for testing
├── champions/              # Frozen versions: v6_2, v7, v8, v9, v10
├── tools/                  # Build, test and benchmark scripts (see below)
├── reports/                # Benchmark reports
└── docs/                   # Official problem statement, rulebook, protocol
```

## How the bot plays

1. **Preflop:** 3-card hand strength from `preflop_table.py`, with position-aware opening, defending and 3-bet thresholds adjusted to the opponent's measured frequencies.
2. **Discard:** Monte-Carlo equity of each 2-card keep against the opponent's likely holdings on the flop.
3. **Opponent range model:** once the opponent discards, all 3 of their starting cards are known for each candidate hand. Each candidate is weighted by how consistent it is with their discard and with their preflop line (open, 3-bet, call, limp), using their own measured frequencies.
4. **Equity:** exact enumeration of every opponent hand on the river, and Monte-Carlo on the flop and turn, scaled to the time bank.
5. **Facing bets:**
   - Very large turn/river bets (1.1× pot or more) are compared against the strongest slice of the opponent's range, with the slice size and bluff share learned per opponent.
   - Small bets (34% of the pot or less) are decided on price: equity against pot odds.
6. **Safety:** every message handler is wrapped in try/except, every raise is clamped to the legal range, and errors go to stderr only. A bug costs at most one hand, never the match.

## Version history (each measured head-to-head on fresh deals)

| Version | Change | Result |
| --- | --- | --- |
| v6 | Exact river equity, time-scaled simulations | +30 ± 14 chips/match vs v5 (300 matches) |
| v6.2 | Two-leg format fixes, tuning | +71 ± 27 vs v6 (120) |
| v7 | Opponent hands weighted by their preflop line | +48 ± 10 vs v6.2 (540) |
| v8 | Lookup-table evaluator, turn/river fixes | +82 ± 22 vs v7 (160) |
| v9 | Learned defence against very large bets | vs spar bot: +391 vs +139 for v8 (100); +27 ± 12 vs v8 (200) |
| v10 | Price floor for small bets | vs min-bet bot: +233 vs −57 for v9 (30); +9 ± 13 vs v9 (200) |

## Robustness (v10)

- 10,000 fuzzed game states: 0 crashes, 0 illegal replies.
- Protocol edge cases: all pass (all-ins, raise cap, junk input, missing fields, 40 ms time bank, clean exit).
- Chaos opponent sending illegal moves and junk: 0 notices.
- At most about 4 s of CPU per 500-hand leg (75 s allowed); about 34 MB of memory.

## Build and test

Run from the repository root with Python 3.11:

```bash
python3.11 tools/build_bot.py                                   # rebuild Starter_Kit/bot.py from the sources
python3.11 tools/final_check.py                                 # one-command pre-submission check (all must PASS)
python3.11 tools/fuzz.py Starter_Kit/bot.py 5000                # random-state crash test
python3.11 tools/edge_test.py Starter_Kit/bot.py                # protocol edge cases
python3.11 tools/h2h.py Starter_Kit/bot.py champions/v9.py 300  # head-to-head with standard error
python3.11 tools/gauntlet.py Starter_Kit/bot.py --seeds 4       # vs every bot in example_bots/
python3.11 Starter_Kit/engine.py Starter_Kit/bot.py spar/spar_bot.py --stage 2   # single match
```

A change is kept only if `h2h` against the current champion reports **REAL improvement** and `final_check.py` passes.
