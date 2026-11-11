# Official Tournament Benchmark Report: `bot.py` vs `spar_bot.py`

**Date / Context**: Official Two-Leg Match Runner (`engine.py`, `LEGS = 2`, duplicate deals swapped in Leg 2).
- **Bot A (Hero)**: `bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/Starter_Kit/bot.py`)
- **Bot B (Opponent)**: `spar_bot.py` (`/Users/pragunkk/Library/CloudStorage/OneDrive-iiit-b/Hackathons/pokerBots/spar/spar_bot.py`)

## 1. Stage 1 Match-by-Match Standings (400 Hands / Match, Cap $C = 400$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `1001` | B (spar_bot.py) | `-506` | `-400` |  2.76s /  6.09s | 0 / 0 |
| Match #2 | `1102` | B (spar_bot.py) | `-586` | `-400` |  2.97s /  6.03s | 0 / 0 |
| Match #3 | `1203` | **A (bot.py)** | `+97` | `+97` |  2.69s /  5.77s | 0 / 0 |
| Match #4 | `1304` | **A (bot.py)** | `+389` | `+389` |  2.88s /  6.29s | 0 / 0 |
| Match #5 | `1405` | B (spar_bot.py) | `-578` | `-400` |  2.62s /  5.69s | 0 / 0 |
| **Stage 1 Total** | **5 Matches** | **2.0 / 5 (2W-0D-3L)** | **`-1184` (-0.592/h)** | **Score_cap = `-0.3570`** | **—** | **0 / 0** |

## 2. Stage 2 Finals Standings (1,000 Hands / Match, Cap $C = 1000$)
| Match # | Seed | Winner | Raw Net (A) | Capped Delta (A) | Time A / B (2 Legs) | Notices A / B |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Match #1 | `777` | B (spar_bot.py) | `-785` | `-785` | 10.04s / 21.38s | 0 / 0 |
| Match #2 | `888` | B (spar_bot.py) | `-755` | `-755` | 10.92s / 22.79s | 0 / 0 |
| **Stage 2 Total** | **2 Matches** | **0.0 / 2 (0W-0D-2L)** | **`-1540` (-0.770/h)** | **Score_cap = `-0.7700`** | **—** | **0 / 0** |

## 3. Overall Tournament Aggregate Summary
- **Total Hands Played**: `4000` hands across `7` matches
- **Match Points**: **`2.0 / 7` (28.6% Win Rate)**
- **Official Score_cap**: **`-0.5635` / hand**
- **Total Raw Net Chips**: **`-2724` chips (`-0.681` chips/hand)**

## 4. Deep Hand-Level & Street-by-Street Forensics
| Metric Category | Hero: `bot.py` (A) | Opponent: `spar_bot.py` (B) | Strategic Insight |
| :--- | :---: | :---: | :--- |
| **Button Open Frequency** | `81.2%` (1624/2000) | `61.3%` (1225/2000) | Preflop steal aggression |
| **Fold to Preflop Open (BB)** | `28.8%` (353/1225) | `10.9%` (177/1624) | Big Blind defense tightness |
| **Preflop 3-Bet Frequency** | `188` times | `306` times | Preflop re-raise pressure |
| **Total Folds Across Hands** | `1664` (41.6%) | `1644` (41.1%) | Overall fold frequency |
| **Preflop Folds** | `732` (18.3%) | `594` (14.8%) | Hand surrendered preflop |
| **Flop Folds** | `160` | `632` | C-bet / probe fold frequency |
| **Turn Folds** | `499` | `196` | Turn barrel fold frequency |
| **River Folds** | `273` | `222` | River fold frequency |
| **Showdown Reach Frequency** | `692` hands (17.3%) | `692` hands (17.3%) | Hands reaching final showdown |
| **Showdowns Won** | `325` (47.0%) | `314` (45.4%) | Showdown equity conversion |
| **Big Pots Won (>= 40 chips)** | **`125`** pots | **`127`** pots | High-leverage stack-off edge |
