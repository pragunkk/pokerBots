# Official Tournament Benchmark Report: `rl_bot/bot.py` vs `bot_panav.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `rl_bot/bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/rl_bot/bot.py`)
- **Bot B (Opponent)**: `bot_panav.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/opponentBots/bot_panav.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | **A (rl_bot/bot.py)** | `+186` | `+186` |  0.00s /  0.00s | 0 / 0 |
| Match #2 | `1102` | **A (rl_bot/bot.py)** | `+228` | `+228` |  0.00s /  0.00s | 0 / 0 |
| Match #3 | `1203` | B (bot_panav.py) | `-391` | `-391` |  0.00s /  0.00s | 0 / 0 |
| Match #4 | `1304` | B (bot_panav.py) | `-60` | `-60` |  0.00s /  0.00s | 0 / 0 |
| Match #5 | `1405` | B (bot_panav.py) | `-317` | `-317` |  0.00s /  0.00s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **2.0 / 5 (2W-0D-3L)** | **`-354` (-0.177/h)** | **Score_cap = `-0.1770`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | B (bot_panav.py) | `-1457` | `-1000` |  0.65s /  2.59s | 0 / 0 |
| Match #2 | `888` | B (bot_panav.py) | `-1252` | `-1000` |  0.73s /  3.19s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **0.0 / 2 (0W-0D-2L)** | **`-2709` (-1.355/h)** | **Score_cap = `-1.0000`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`2.0 / 7` (28.6% Win Rate)**
- **Official Score_cap**: **`-0.5885` / hand**
- **Total Raw Net Chips**: **`-3063` chips (`-0.766` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `rl_bot/bot.py` (A) | Opponent: `bot_panav.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `51.2%` (1025/2000) | `13.6%` (271/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `29.9%` (81/271) | `7.5%` (77/1025) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `26` times | `38` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `1926` (48.1%) | `1242` (31.1%) | Overall fold frequency |
| **Preflop Folds** | `276` (6.9%) | `96` (2.4%) | Hand surrendered preflop |
| **Flop Folds** | `902` | `611` | C-bet / probe fold frequency |
| **Turn Folds** | `506` | `307` | Turn barrel fold frequency |
| **River Folds** | `242` | `228` | River fold frequency |
| **Showdown Reach Frequency** | `832` hands (20.8%) | `832` hands (20.8%) | Hands reaching final showdown |
| **Showdowns Won** | `246` (29.6%) | `555` (66.7%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`127`** pots | **`143`** pots | High-leverage stack-off edge |
