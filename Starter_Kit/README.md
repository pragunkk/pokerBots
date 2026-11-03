# Pokerbots Starter Kit & Championship Test Suite: Discard Hold'em

| File | Description |
|---|---|
| `bot.py` | **Standalone Championship Submission Bot (Adaptive Meta-Bot v5)** — self-contained single-file submission (`96.9 KB`, stdlib only, zero file I/O). |
| `adaptive_bot.py` | Modular source for **Adaptive Meta-Bot v5** (`OpponentProfile` dual-window profiler, 5-gear strategy selector, and match-point variance controller). |
| `fast_eval.py` | Sorting-network 5/7-card evaluator, `hole_interaction_level`, `board_danger_level`, flop discard optimizer, and Bayesian turn/river equity engine. |
| `preflop_table.py` | Embedded lookup table for all **1,755 canonical 3-card preflop isomorphism classes**. |
| `tournament_sim.py` | Parallel multi-seed tournament harness (`proven`, `heuristics`, and `full20` suites). |
| `engine.py` | Official local match runner (`--stage 1` for 400 hands, `--stage 2` for 1000 hands). |
| `example_bots/` | **20-Bot Opponent Suite**:<br>• **Extreme Heuristics**: `always_fold_bot.py`, `always_call_bot.py`, `always_shove_bot.py`, `always_minraise_bot.py`, `always_potbet_bot.py`<br>• **Proven Poker Algorithms**: `cfr_nash_bot.py`, `ehu_sankey_bot.py`, `kelly_pot_geometric_bot.py`, `bayesian_range_bot.py`, `uchicago_slumbot_hybrid.py`<br>• **Archetypes & Rivals**: `starter_bot.py`, `random_bot.py`, `calling_station_bot.py`, `tag_bot.py`, `lag_maniac_bot.py`, `trapper_bot.py`, `cfr_gto_bot.py`, `rival_adaptive_bot.py`<br>• **Self-Play Evolutionary Mirrors**: `adaptive_v3_mirror.py`, `adaptive_v4_mirror.py` |

## Quick Start

```bash
# Play a 400-hand Stage 1 match against any bot
python3 engine.py bot.py example_bots/cfr_nash_bot.py

# Play a 1000-hand Stage 2 match
python3 engine.py bot.py example_bots/adaptive_v4_mirror.py --stage 2

# Run parallel multi-seed benchmark suites
python3 tournament_sim.py proven bot.py
python3 tournament_sim.py heuristics bot.py
python3 tournament_sim.py full20 bot.py
```

## Adaptive Bot v5 Highlights

- **27–0–1 (`96.4%` Match Points, `+16,621` Net Chips)** across 4 seeds against the 5 Proven Computer-Poker Algorithms (`CFR Nash`, `UAlberta EHS/EHS^2`, `DeepStack/Pluribus Geometric/Kelly`, `Bayesian Combinatorial Range`, `Tartanian/Libratus Hybrid`) and both previous generations (`Adaptive v4 Mirror` `4–0–0` and `Adaptive v3 Mirror` `4–0–0`).
- **20–0–0 (`100.0%` Match Points, `+71,410` Net Chips)** across 4 seeds against the 5 Extreme Heuristics (`+450/450` theoretical maximum vs. `Always Fold`, `+13,524` vs. `Always Shove`, `+22,067` vs. `Always Min-Raise`, `+19,988` vs. `Always Pot-Bet`, `+14,031` vs. `Always Call`).
