# Official Tournament Benchmark Report: `rl_bot/bot.py` vs `bot_panav.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `rl_bot/bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/rl_bot/bot.py`)
- **Bot B (Opponent)**: `bot_panav.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/opponentBots/bot_panav.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | B (bot_panav.py) | `-114` | `-114` |  0.00s /  0.00s | 0 / 0 |
| Match #2 | `1102` | B (bot_panav.py) | `-147` | `-147` |  0.00s /  0.00s | 0 / 0 |
| Match #3 | `1203` | **A (rl_bot/bot.py)** | `+313` | `+313` |  0.00s /  0.00s | 0 / 0 |
| Match #4 | `1304` | B (bot_panav.py) | `-468` | `-400` |  0.00s /  0.00s | 0 / 0 |
| Match #5 | `1405` | B (bot_panav.py) | `-504` | `-400` |  0.00s /  0.00s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **1.0 / 5 (1W-0D-4L)** | **`-920` (-0.460/h)** | **Score_cap = `-0.3740`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | B (bot_panav.py) | `-1677` | `-1000` |  0.82s /  3.22s | 0 / 0 |
| Match #2 | `888` | B (bot_panav.py) | `-1197` | `-1000` |  0.86s /  3.81s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **0.0 / 2 (0W-0D-2L)** | **`-2874` (-1.437/h)** | **Score_cap = `-1.0000`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`1.0 / 7` (14.3% Win Rate)**
- **Official Score_cap**: **`-0.6870` / hand**
- **Total Raw Net Chips**: **`-3794` chips (`-0.949` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `rl_bot/bot.py` (A) | Opponent: `bot_panav.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `51.2%` (1025/2000) | `14.7%` (294/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `30.6%` (90/294) | `4.3%` (44/1025) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `30` times | `43` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `1908` (47.7%) | `1214` (30.3%) | Overall fold frequency |
| **Preflop Folds** | `294` (7.3%) | `71` (1.8%) | Hand surrendered preflop |
| **Flop Folds** | `878` | `589` | C-bet / probe fold frequency |
| **Turn Folds** | `498` | `309` | Turn barrel fold frequency |
| **River Folds** | `238` | `245` | River fold frequency |
| **Showdown Reach Frequency** | `878` hands (21.9%) | `878` hands (21.9%) | Hands reaching final showdown |
| **Showdowns Won** | `276` (31.4%) | `566` (64.5%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`152`** pots | **`147`** pots | High-leverage stack-off edge |
