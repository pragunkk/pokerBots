# Official Tournament Benchmark Report: `bot.py` vs `spar_bot.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/Starter_Kit/bot.py`)
- **Bot B (Opponent)**: `spar_bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/spar/spar_bot.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | **A (bot.py)** | `+689` | `+400` |  0.48s /  3.89s | 0 / 0 |
| Match #2 | `1102` | B (spar_bot.py) | `-674` | `-400` |  0.40s /  3.16s | 0 / 0 |
| Match #3 | `1203` | B (spar_bot.py) | `-72` | `-72` |  0.37s /  2.97s | 0 / 0 |
| Match #4 | `1304` | B (spar_bot.py) | `-63` | `-63` |  0.41s /  3.17s | 0 / 0 |
| Match #5 | `1405` | **A (bot.py)** | `+172` | `+172` |  0.40s /  3.18s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **2.0 / 5 (2W-0D-3L)** | **`+52` (+0.026/h)** | **Score_cap = `+0.0185`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | B (spar_bot.py) | `-183` | `-183` |  2.27s / 18.33s | 0 / 0 |
| Match #2 | `888` | **A (bot.py)** | `+202` | `+202` |  2.45s / 19.50s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **1.0 / 2 (1W-0D-1L)** | **`+19` (+0.009/h)** | **Score_cap = `+0.0095`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`3.0 / 7` (42.9% Win Rate)**
- **Official Score_cap**: **`+0.0140` / hand**
- **Total Raw Net Chips**: **`+71` chips (`+0.018` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `bot.py` (A) | Opponent: `spar_bot.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `75.0%` (1499/2000) | `61.2%` (1223/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `27.5%` (336/1223) | `12.6%` (189/1499) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `218` times | `251` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `1749` (43.7%) | `1603` (40.1%) | Overall fold frequency |
| **Preflop Folds** | `726` (18.1%) | `625` (15.6%) | Hand surrendered preflop |
| **Flop Folds** | `343` | `582` | C-bet / probe fold frequency |
| **Turn Folds** | `385` | `191` | Turn barrel fold frequency |
| **River Folds** | `295` | `205` | River fold frequency |
| **Showdown Reach Frequency** | `648` hands (16.2%) | `648` hands (16.2%) | Hands reaching final showdown |
| **Showdowns Won** | `325` (50.2%) | `274` (42.3%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`133`** pots | **`91`** pots | High-leverage stack-off edge |
