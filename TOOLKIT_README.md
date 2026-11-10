# Pokerbots toolkit

Unzip into the repo root (next to `README.md` and `Starter_Kit/`). Run every command from the repo root.

| Command | Purpose |
| --- | --- |
| `python tools/build_bot.py` | Build `Starter_Kit/bot.py` from `preflop_table.py` + `fast_eval.py` + `adaptive_bot.py` |
| `python tools/fuzz.py Starter_Kit/bot.py 5000` | Random game states: crashes, illegal replies, slowest reply |
| `python tools/h2h.py NEW.py OLD.py 100` | New vs old over 100 seeds, with standard error and a verdict |
| `python tools/gauntlet.py Starter_Kit/bot.py --seeds 4` | One bot vs every bot in `Starter_Kit/example_bots/` |
| `python tools/gen_preflop.py --samples 20000` | Regenerate the preflop table (experimental, see the playbook step 5) |
| `tools/safe_main.py` | Crash-proof main loop: paste over `main()` in `adaptive_bot.py` |

New sparring bots in `Starter_Kit/example_bots/`: `pushfold_wide_bot.py`, `pushfold_tight_bot.py`, `plain_equity_bot.py`.

Rule: edit the three source files, never `bot.py`; merge a change only when `h2h` says REAL improvement.
