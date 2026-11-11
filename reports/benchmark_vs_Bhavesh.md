# Official Tournament Benchmark Report: `bot.py` vs `bot_Bhavesh.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/Starter_Kit/bot.py`)
- **Bot B (Opponent)**: `bot_Bhavesh.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/opponentBots/bot_Bhavesh.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | **A (bot.py)** | `+1073` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #2 | `1102` | **A (bot.py)** | `+1000` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #3 | `1203` | **A (bot.py)** | `+905` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #4 | `1304` | **A (bot.py)** | `+1008` | `+400` |  0.00s /  0.00s | 0 / 0 |
| Match #5 | `1405` | **A (bot.py)** | `+1024` | `+400` |  0.00s /  0.00s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **5.0 / 5 (5W-0D-0L)** | **`+5010` (+2.505/h)** | **Score_cap = `+1.0000`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | **A (bot.py)** | `+2530` | `+1000` |  0.00s /  0.00s | 0 / 0 |
| Match #2 | `888` | **A (bot.py)** | `+2761` | `+1000` |  0.00s /  0.00s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **2.0 / 2 (2W-0D-0L)** | **`+5291` (+2.646/h)** | **Score_cap = `+1.0000`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`7.0 / 7` (100.0% Win Rate)**
- **Official Score_cap**: **`+1.0000` / hand**
- **Total Raw Net Chips**: **`+10301` chips (`+2.575` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `bot.py` (A) | Opponent: `bot_Bhavesh.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `99.6%` (1992/2000) | `83.5%` (1670/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `26.8%` (447/1670) | `100.0%` (1992/1992) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `330` times | `0` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `449` (11.2%) | `3483` (87.1%) | Overall fold frequency |
| **Preflop Folds** | `449` (11.2%) | `2652` (66.3%) | Hand surrendered preflop |
| **Flop Folds** | `0` | `757` | C-bet / probe fold frequency |
| **Turn Folds** | `0` | `35` | Turn barrel fold frequency |
| **River Folds** | `0` | `39` | River fold frequency |
| **Showdown Reach Frequency** | `68` hands (1.7%) | `68` hands (1.7%) | Hands reaching final showdown |
| **Showdowns Won** | `28` (41.2%) | `39` (57.4%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`0`** pots | **`0`** pots | High-leverage stack-off edge |
