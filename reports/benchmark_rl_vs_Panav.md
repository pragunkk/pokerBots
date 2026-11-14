# Official Tournament Benchmark Report: `rl_bot/bot.py` vs `bot_panav.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `rl_bot/bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/rl_bot/bot.py`)
- **Bot B (Opponent)**: `bot_panav.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/opponentBots/bot_panav.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | **A (rl_bot/bot.py)** | `+831` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #2 | `1102` | **A (rl_bot/bot.py)** | `+484` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #3 | `1203` | **A (rl_bot/bot.py)** | `+711` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #4 | `1304` | **A (rl_bot/bot.py)** | `+641` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #5 | `1405` | **A (rl_bot/bot.py)** | `+558` | `+400` |  0.00s /  0.00s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **5.0 / 5 (5W-0D-0L)** | **`+3225` (+1.613/h)** | **Score_cap = `+1.0000`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | B (bot_panav.py) | `-1869` | `-1000` |  0.53s /  1.90s | 0 / 0 |
| Match #2 | `888` | B (bot_panav.py) | `-574` | `-574` |  0.57s /  2.25s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **0.0 / 2 (0W-0D-2L)** | **`-2443` (-1.222/h)** | **Score_cap = `-0.7870`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`5.0 / 7` (71.4% Win Rate)**
- **Official Score_cap**: **`+0.1065` / hand**
- **Total Raw Net Chips**: **`+782` chips (`+0.196` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `rl_bot/bot.py` (A) | Opponent: `bot_panav.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `51.0%` (1020/2000) | `18.4%` (369/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `32.0%` (118/369) | `6.5%` (66/1020) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `42` times | `40` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `1499` (37.5%) | `1878` (46.9%) | Overall fold frequency |
| **Preflop Folds** | `349` (8.7%) | `91` (2.3%) | Hand surrendered preflop |
| **Flop Folds** | `574` | `1352` | C-bet / probe fold frequency |
| **Turn Folds** | `390` | `252` | Turn barrel fold frequency |
| **River Folds** | `186` | `183` | River fold frequency |
| **Showdown Reach Frequency** | `623` hands (15.6%) | `623` hands (15.6%) | Hands reaching final showdown |
| **Showdowns Won** | `180` (28.9%) | `417` (66.9%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`150`** pots | **`141`** pots | High-leverage stack-off edge |
