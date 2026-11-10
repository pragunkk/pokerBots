# Pokerbots Hackathon: Discard Hold'em — Adaptive Meta-Bot v6.1 & Official Two-Leg Tournament Runner

Championship submission and 24-bot testing framework for **Discard Hold'em** (Python 3.11, protocol v1.0), updated for the organisers' **Two-Leg Match Format (`LEGS = 2`)**.

## 1. Summary of Organiser Updates (`update/` Folder)

| Area | Old Specification (Nov 1) | Updated Official Specification (Nov 4) | Implementation in Our Project |
| :--- | :--- | :--- | :--- |
| **Match Format (`LEGS = 2`)** | Single continuous process playing 400 hands (Stage 1) or 1000 hands (Stage 2), with duplicate deals on consecutive hand IDs `(2k+1, 2k+2)`. | **Two Legs per Match (`LEGS = 2`)**: Stage 1 = 2 legs of `200` deals (`400` hands total); Stage 2 = 2 legs of `500` deals (`1000` hands total). Fresh bot processes are launched for each leg (`init` sends `total_hands: 200` or `500`), and Leg 2 swaps hole cards and seats (`button = (deal + leg) % 2`). | Updated `Starter_Kit/engine.py`, `Starter_Kit/adaptive_bot.py`, and `Starter_Kit/bot.py` (`v6.1`). Removed single-leg `PROTECT_WIN`/`COMEBACK` throttling because card luck only cancels across Leg 1 + Leg 2 combined—maximizing chip EV on every hand in both legs is strictly optimal. |
| **Per-Leg Time Bank & Wall Clock** | `60s` time bank / `180s` wall-clock (Stage 1); `150s` time bank / `450s` wall-clock (Stage 2). | **`30s` time bank / `90s` wall-clock per leg** in Stage 1; **`75s` time bank / `225s` wall-clock per leg** in Stage 2 (`5.0s` free start-up per leg). | Calibrated rollout thresholds in `adaptive_bot.py` and `bot.py` to the `30,000 ms` per-leg bank (`> 8,000 ms` for full 90-rollout flop, 145-rollout turn, and 903-hand exact river enumeration; `< 2,000 ms` instant guard). |
| **Logs & Forfeits** | Single `A_stderr.txt`, `B_stderr.txt`; hand logs `#1..#400`. | Per-leg stderr files (`A_leg1_stderr.txt`, `A_leg2_stderr.txt`, `B_leg1_stderr.txt`, `B_leg2_stderr.txt`) and leg-prefixed hand logs (`L1 #1`, `L2 #1`). A forfeit in Leg 1 skips Leg 2 and awards `+C` / `-C`. | Integrated into `Starter_Kit/engine.py` and parsed automatically in `Starter_Kit/run_tournament.py`. |

## 2. How to Run Practice Matches & Tournaments Against Your Friend's Bot

All tournament and match runners are inside `Starter_Kit/`:

```bash
cd Starter_Kit
```

### A. Single Official Two-Leg Match (`engine.py`)
Run a single Stage 1 (`400` hands = 2 legs of `200` deals) or Stage 2 (`1000` hands = 2 legs of `500` deals) match between our `bot.py` and your friend's bot:

```bash
# Stage 1 match (400 hands across 2 legs, 30s bank/leg, Cap C = 400)
python3 engine.py bot.py /path/to/friend_bot.py

# Stage 2 Finals match (1000 hands across 2 legs, 75s bank/leg, Cap C = 1000)
python3 engine.py bot.py /path/to/friend_bot.py --stage 2

# Replay a specific deal seed and print every hand live to the console
python3 engine.py bot.py /path/to/friend_bot.py --seed 42 --verbose
```
- Hand-by-hand logs (`L1 #1 ...`, `L2 #1 ...`) are written to `Starter_Kit/logs/match.log`.

### B. Multi-Match Head-to-Head Practice Series (`run_tournament.py`)
To get statistically reliable results across multiple random seeds (with automatic parallel execution, Match Points, official Capped Score $\text{Score}_{\text{cap}}$, and raw chip averages):

```bash
# Play 5 Stage-1 matches (2,000 hands total) in parallel against your friend's bot
python3 run_tournament.py bot.py /path/to/friend_bot.py --stage 1 --matches 5

# Play 5 Stage-2 Finals matches (5,000 hands total) in parallel
python3 run_tournament.py bot.py /path/to/friend_bot.py --stage 2 --matches 5

# Save match logs for every match in the series
python3 run_tournament.py bot.py /path/to/friend_bot.py --matches 5 --log-prefix logs/friend_series
```

### C. Full Round-Robin or Swiss Tournament (`run_tournament.py`)
Include your friend's bot alongside our `bot.py` and any of the 24 bots in `example_bots/` to simulate an official Stage 1 Swiss Qualifier or Stage 2 Round-Robin Finals table (ranked by **Match Points**, then **Official $\text{Score}_{\text{cap}} = \frac{\sum \text{clip}(\Delta_m, -C, C)}{\sum H_m}$**):

```bash
# Official Round-Robin Tournament
python3 run_tournament.py --round-robin \
  bot.py \
  /path/to/friend_bot.py \
  example_bots/cfr_nash_bot.py \
  example_bots/exact_bayesian_ev_bot.py \
  example_bots/kelly_pot_geometric_bot.py \
  example_bots/lag_maniac_bot.py

# Official Stage 1 Swiss Qualifier Tournament (e.g., 6 rounds)
python3 run_tournament.py --swiss --rounds 6 \
  bot.py \
  /path/to/friend_bot.py \
  example_bots/cfr_nash_bot.py \
  example_bots/exact_bayesian_ev_bot.py \
  example_bots/kelly_pot_geometric_bot.py \
  example_bots/bayesian_range_bot.py \
  example_bots/tag_bot.py \
  example_bots/trapper_bot.py
```

## 3. Repository Layout

| Path | Description |
| :--- | :--- |
| `Starter_Kit/bot.py` | **Standalone Championship Submission Bot (Adaptive Meta-Bot v6.1)** — self-contained single-file bundle (`97.7 KB`, standard library only, zero runtime file I/O). |
| `Starter_Kit/run_tournament.py` | **Official Two-Leg Practice Series, Round-Robin & Swiss Tournament Runner**. |
| `Starter_Kit/engine.py` | Official updated Two-Leg Discard Hold'em match engine (`LEGS = 2`). |
| `Starter_Kit/adaptive_bot.py` | Modular source for **Adaptive Meta-Bot v6.1**. |
| `Starter_Kit/fast_eval.py` | Sorting-network 5/7-card hand evaluator, **Full 903-Hand Exact Combinatorial River Solver (`exact_river_equity`)**, and **Pre-Evaluated Turn Rollout Engine (`145` rollouts)**. |
| `Starter_Kit/preflop_table.py` | Embedded lookup table for all **1,755 canonical 3-card preflop isomorphism classes**. |
| `Starter_Kit/tournament_sim.py` | 24-bot benchmark suite runner (`proven`, `heuristics`, `full20`). |
| `problemStatement.pdf`, `ruleBook.pdf`, `protocol.pdf` | Updated official tournament documents (Nov 4 revision). |
