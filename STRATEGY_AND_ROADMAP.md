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
   - Because each player chooses the best 2 of 3 hole cards *after* seeing the 3 flop cards, average hand strength at the flop and showdown is substantially higher than in standard 2-card Texas Hold'em.

2. **Simultaneous Exposed Post-Flop Discard**:
   - After flop betting concludes (if neither player folded, including when all-in), both players simultaneously discard 1 hole card face up (`discard_result` reveals both `my_discard` and `opp_discard`).
   - **Card Removal (Dead Card Blockers)**: `my_discard` and `opp_discard` can **never** appear on the Turn or River and are not in the opponent's remaining 2-card hand.
   - **Bayesian Signaling & Range Narrowing**: Observing `opp_discard` leaks strong information about `opp_hole_cards` (high-card discard signals made hands or strong draws; suited discard on a 2-flush flop discounts opponent flush draws).
   - **Deceptive Discards**: When two discard candidates are within $\le 1.0\%$ equity, our bot discards the **lower** card when holding a monster (disguising strength) and discards the **higher** redundant card when semi-bluffing.

---

## 2. Roadmap & Checkpoints Summary

- **Checkpoint 1 (Completed)**: Fast sorting-network 5/7-card hand evaluator, rollout discard optimizer, and parallel tournament simulator.
- **Checkpoint 2 (Completed)**: 6-bot Opponent Archetype Zoo and Adaptive Bot v1.
- **Checkpoint 3 (Completed)**: Exact 1,755 canonical 3-card preflop lookup table (`preflop_table.py`), board-relative hand interaction filter (`hole_interaction_level`), and structural post-flop raise gating.
- **Checkpoint 4 (Completed)**: Dual-Window Opponent Profiler (Global + 50-Hand Sliding Window), strict `MANIAC` vs `NIT_TAG` separation, and `24-0-0` Stage 1 sweep (`+52,036` net chips).
- **Checkpoint 5 (Completed)**: Proven Computer-Poker Algorithm Suite (`cfr_nash_bot.py`, `ehu_sankey_bot.py`, `kelly_pot_geometric_bot.py`, `bayesian_range_bot.py`, `uchicago_slumbot_hybrid.py`), Self-Play v3 Mirror testing, and **Adaptive Bot v4** (`39-0-3`, `92.9%` across 14 bots, `+52,360` net chips).
