# Pokerbots Hackathon: Discard Hold'em — Deep Analysis, Winning Strategy & Roadmap

## 1. Deep Analysis of the Game Rules & Mathematical Mechanics

### 1.1 Game Parameters & Structure
- **Format**: Heads-Up (2-player) No-Limit **Discard Hold'em**.
- **Blinds & Stacks**: Small Blind (Button) = `1`, Big Blind = `2`, Starting Stack = `200` chips (`100 BB`), **reset every hand**.
- **Betting Rules**:
  - Preflop: Button (SB) acts first; BB acts second.
  - Postflop (Flop, Turn, River): Big Blind acts first; Button acts second.
  - Raise amounts are **"to" amounts** (total street commitment).
  - Minimum raise-to: `opp_bet + max(last_raise_size, 2)`.
  - **4-Bet Cap per Street**: At most 4 bets/raises per street (the Big Blind counts as the 1st preflop, so at most 3 raises preflop: 2-bet, 3-bet, 4-bet).
- **Duplicate Deals**:
  - Every deal is played twice with seat cards swapped against the identical 5-card board (`200` duplicate deals = `400` hands in Stage 1; `500` duplicate deals = `1000` hands in Stage 2).
  - Duplicate deals cancel out raw card-distribution luck and heavily reward superior post-flop discard selection, positional extraction, and opponent exploitation.

### 1.2 The Two Core Variant Mechanics (Where Edges Are Won)

1. **Three Hole Cards Through the Flop Betting Round**:
   - Each player receives **3 hole cards** ($\binom{52}{3} = 22,100$ combinations; $1,755$ strategically distinct isomorphism classes).
   - Crucially, **both Preflop AND Flop betting happen before the discard!**
   - Because each player chooses the best 2 of 3 hole cards *after* seeing the 3 flop cards, average hand strength at the flop and showdown is substantially higher than in standard 2-card Texas Hold'em:
     - A 3-card hand contains $\binom{3}{2} = 3$ distinct two-card Hold'em hands simultaneously.
     - Hands like `As Ks Td` or `9s 9h Qc` have massive post-flop flexibility: if the flop hits the high card (`Q`), you keep top pair or make a set; if the flop misses, you keep the pocket pair (`99`); if two spades flop, you keep the flush draw.
     - **Three-of-a-kind in the hole** (e.g., `Ks Kh Kd`) is paradoxically weaker than a Pocket Pair + High Suited Kicker (`Ks Kh As`) because you are **forced to discard one of your three cards** after the flop AND you block the board from giving you a set!

2. **Simultaneous Exposed Post-Flop Discard**:
   - After flop betting concludes (if neither player folded, including when all-in), both players simultaneously discard 1 hole card face up (`discard_result` reveals both `my_discard` and `opp_discard`).
   - **Card Removal (Dead Card Blockers)**:
     - The board is fixed before the hand, and discarded cards were in the players' hands—meaning `my_discard` and `opp_discard` can **never** appear on the Turn or River and are not in the opponent's remaining 2-card hand.
   - **Bayesian Signaling & Range Narrowing**:
     - A rational opponent discards the card $d \in \{c_1, c_2, c_3\}$ that minimizes their remaining 2-card equity on the flop:
       $$d^* = \arg\max_{d \in H_3} \text{Equity}(H_3 \setminus \{d\} \mid \text{Flop})$$
     - Therefore, observing `opp_discard` leaks massive information about `opp_hole_cards`:
       - **High-Card Discard Signal**: If the opponent discards `Kc` or `Ad` on a `Th 8h 3c` board, their remaining two cards were strictly preferable to keeping an overcard `K`/`A`. That sharply narrows their range to made hands (top pair+, overpair, two-pair, set) or monster combo/flush draws.
       - **Suit/Flush-Draw Dismissal**: If the flop has two hearts (`Th 8h 3c`) and the opponent discards a heart (`6h`), they almost certainly do **not** hold a heart flush draw (unless they started with 3 hearts, which is rare).
       - **Deceptive Discards (Counter-Exploitation)**: Whenever our bot has two discard choices within $\le 0.8\%$ equity (e.g., holding a pocket pair or set with two useless kickers), we deliberately discard the **lower** card when holding a monster (disguising our strength as a routine trash discard) and discard the **higher** redundant card when holding a medium/bluffing hand (representing strength to fold out observant opponents).

### 1.3 Tournament Structure & Scoring Game Theory
- **Stage 1 (Swiss)**: 6–8 rounds, `400` hands/match, `60s` time bank (`150 ms/hand`), Cap $C = 400$. Top 32 qualify.
- **Stage 2 (Finals Round-Robin)**: Top 32 play all 31 opponents (`496` matches total), `1000` hands/match, `150s` time bank (`150 ms/hand`), Cap $C = 1000$.
- **Primary Ranking Metric = Match Points** (`1.0` Win, `0.5` Draw, `0.0` Loss), with **Capped Chips/Hand** ($\text{clip}(\Delta_m, -C, +C)$) as the tie-breaker.
- **Strategic Implication**:
  - Winning a match by $+1200$ yields the exact same match points (`1.0`) and capped tie-breaker (`+400` in Stage 1, `+1000` in Stage 2) as winning by $+C$.
  - Therefore, our bot includes a **Match-Point & Cap Variance Manager**:
    - **Lead-Protection Mode (`PROTECT_WIN`)**: When comfortably ahead late in a match or above $+C + 140$, tighten high-variance all-in calls and reduce bluff frequencies to lock in the `1.0` match point and maximum $+C$ tie-break score.
    - **Comeback Mode (`COMEBACK`)**: When trailing late in a match, increase aggression and variance to maximize $P(\Delta_{\text{final}} > 0)$.

---

## 2. Roadmap & Checkpoints

### Checkpoint 1: Core Fast Evaluation & Simulation Infrastructure (COMPLETED)
- [x] Build bitmask/sorting-network 5-card & 7-card evaluator (`Starter_Kit/fast_eval.py`).
- [x] Build rollout-based Flop Discard Optimizer (`evaluate_discard_options`) and Bayesian Turn/River Equity Calculator (`post_discard_equity`).
- [x] Build parallel multi-seed tournament simulator (`Starter_Kit/tournament_sim.py`).

### Checkpoint 2: Opponent Archetype Zoo & Adaptive Meta-Bot v1 (COMPLETED)
- [x] Create diverse opponent archetypes in `Starter_Kit/example_bots/`:
  - `calling_station_bot.py` (Loose-Passive)
  - `tag_bot.py` (Tight-Aggressive)
  - `lag_maniac_bot.py` (Loose-Aggressive Bully)
  - `trapper_bot.py` (Slow-play / Check-Raise + Deceptive Discards)
- [x] Build `Starter_Kit/adaptive_bot.py` and run initial benchmark (identified multi-street Maniac barrel leak).

### Checkpoint 3: Exact 1,755 Pre-Flop Lookup Table & Structural Post-Flop Gating (COMPLETED)
- [x] Precompute all **1,755 canonical 3-card preflop isomorphism classes** (under post-flop optimal discard rules) into `Starter_Kit/preflop_table.py` for instant `O(1)` lookup without filesystem reads.
- [x] Implement `hole_interaction_level` in `Starter_Kit/fast_eval.py` to distinguish board-only pairs (`inter = 0`), weak/underpairs (`inter = 1`), Top Pair/Overpair (`inter = 2`), Two Pair/Sets (`inter = 3`), and Straights/Flushes/Full Houses (`inter = 4`).
- [x] Add structural postflop raise-gating (never 3-bet postflop or call big Turn/River raises with mere One Pair against non-maniacs; prevent leading out into the previous street's aggressor without Two Pair+).

### Checkpoint 4: Counter-Meta, Dual-Window Profiler & Rival Meta-Bot Hardening (COMPLETED)
- [x] Build `Starter_Kit/example_bots/cfr_gto_bot.py` (MDF + balanced polarized bet sizing) and `Starter_Kit/example_bots/rival_adaptive_bot.py` (disguised TAG for 40 hands followed by dynamic gear switching).
- [x] Upgrade `OpponentProfile` in `Starter_Kit/adaptive_bot.py` with a **Dual-Window (Global + 50-Hand Sliding Window)** tracker and strict separation of `MANIAC` (`postflop_agg_per_hand > 0.80` + high `PFR`) vs `NIT_TAG` / `BALANCED`.
- [x] Achieve a **24–0–0 (100% Match Win Rate, +52,036 net chips)** sweep across all 8 opponent archetypes in 400-hand Stage 1 matches.

### Checkpoint 5: Full Round-Robin Matrix, Stage-2 (1000-Hand) Stress Testing & Standalone `bot.py` Packaging (NEXT)
- [ ] Run full all-pairs Round-Robin tournament across all bots and Stage-2 (`1000`-hand) stress tests.
- [ ] Package the entire engine (`preflop_table.py`, `fast_eval.py`, `adaptive_bot.py`) into a single self-contained, zero-external-file `Starter_Kit/bot.py` and verify compliance with all submission rules.
