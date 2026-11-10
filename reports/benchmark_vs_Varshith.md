# Official Tournament Benchmark Report: `bot.py` vs `bot_Varshith.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/Starter_Kit/bot.py`)
- **Bot B (Opponent)**: `bot_Varshith.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/opponentBots/bot_Varshith.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | **A (bot.py)** | `+438` | `+400` |  3.63s / 11.61s | 0 / 0 |
| Match #2 | `1102` | B (bot_Varshith.py) | `-354` | `-354` |  3.47s / 10.11s | 0 / 0 |
| Match #3 | `1203` | **A (bot.py)** | `+454` | `+400` |  3.60s / 10.22s | 0 / 0 |
| Match #4 | `1304` | **A (bot.py)** | `+35` | `+35` |  3.97s / 11.59s | 0 / 0 |
| Match #5 | `1405` | **A (bot.py)** | `+330` | `+330` |  3.21s /  9.35s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **4.0 / 5 (4W-0D-1L)** | **`+903` (+0.452/h)** | **Score_cap = `+0.4055`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | **A (bot.py)** | `+61` | `+61` | 11.00s / 31.23s | 0 / 0 |
| Match #2 | `888` | B (bot_Varshith.py) | `-108` | `-108` | 11.78s / 34.41s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **1.0 / 2 (1W-0D-1L)** | **`-47` (-0.024/h)** | **Score_cap = `-0.0235`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`5.0 / 7` (71.4% Win Rate)**
- **Official Score_cap**: **`+0.1910` / hand**
- **Total Raw Net Chips**: **`+856` chips (`+0.214` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `bot.py` (A) | Opponent: `bot_Varshith.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `64.5%` (1289/2000) | `59.5%` (1190/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `29.8%` (355/1190) | `18.5%` (239/1289) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `174` times | `278` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `1740` (43.5%) | `1456` (36.4%) | Overall fold frequency |
| **Preflop Folds** | `760` (19.0%) | `545` (13.6%) | Hand surrendered preflop |
| **Flop Folds** | `271` | `550` | C-bet / probe fold frequency |
| **Turn Folds** | `462` | `154` | Turn barrel fold frequency |
| **River Folds** | `247` | `207` | River fold frequency |
| **Showdown Reach Frequency** | `804` hands (20.1%) | `804` hands (20.1%) | Hands reaching final showdown |
| **Showdowns Won** | `410` (51.0%) | `334` (41.5%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`170`** pots | **`143`** pots | High-leverage stack-off edge |
