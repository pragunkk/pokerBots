# Official Tournament Benchmark Report: `bot.py` vs `bot_Varshith.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/Starter_Kit/bot.py`)
- **Bot B (Opponent)**: `bot_Varshith.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/opponentBots/bot_Varshith.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | **A (bot.py)** | `+404` | `+400` |  0.70s /  8.06s | 0 / 0 |
| Match #2 | `1102` | B (bot_Varshith.py) | `-689` | `-400` |  0.73s /  7.86s | 0 / 0 |
| Match #3 | `1203` | B (bot_Varshith.py) | `-348` | `-348` |  0.76s /  8.05s | 0 / 0 |
| Match #4 | `1304` | **A (bot.py)** | `+360` | `+360` |  0.80s /  9.22s | 0 / 0 |
| Match #5 | `1405` | **A (bot.py)** | `+89` | `+89` |  0.51s /  5.64s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **3.0 / 5 (3W-0D-2L)** | **`-184` (-0.092/h)** | **Score_cap = `+0.0505`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | B (bot_Varshith.py) | `-28` | `-28` |  2.32s / 25.75s | 0 / 0 |
| Match #2 | `888` | **A (bot.py)** | `+616` | `+616` |  2.57s / 30.09s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **1.0 / 2 (1W-0D-1L)** | **`+588` (+0.294/h)** | **Score_cap = `+0.2940`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`4.0 / 7` (57.1% Win Rate)**
- **Official Score_cap**: **`+0.1722` / hand**
- **Total Raw Net Chips**: **`+404` chips (`+0.101` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `bot.py` (A) | Opponent: `bot_Varshith.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `64.5%` (1291/2000) | `59.5%` (1190/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `28.6%` (340/1190) | `19.2%` (248/1291) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `200` times | `263` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `1839` (46.0%) | `1470` (36.8%) | Overall fold frequency |
| **Preflop Folds** | `736` (18.4%) | `563` (14.1%) | Hand surrendered preflop |
| **Flop Folds** | `485` | `544` | C-bet / probe fold frequency |
| **Turn Folds** | `375` | `187` | Turn barrel fold frequency |
| **River Folds** | `243` | `176` | River fold frequency |
| **Showdown Reach Frequency** | `691` hands (17.3%) | `691` hands (17.3%) | Hands reaching final showdown |
| **Showdowns Won** | `359` (52.0%) | `287` (41.5%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`164`** pots | **`137`** pots | High-leverage stack-off edge |
