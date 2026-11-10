#!/usr/bin/env python3
"""Head-to-head A vs B over many seeds, with a standard error so you know if a change is real.

    python tools/h2h.py NEW_BOT OLD_BOT 100 [--hands 400] [--workers 4]

Rule of thumb: a change is real only if |mean| > 2 x s.e.  One 400-hand match has a
standard deviation of roughly 300 chips, so 4 seeds tell you almost nothing."""
import argparse, os, re, statistics, subprocess, sys, concurrent.futures as cf

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Starter_Kit", "engine.py")
p = argparse.ArgumentParser()
p.add_argument("a"); p.add_argument("b"); p.add_argument("n", type=int)
p.add_argument("--hands", default="400"); p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
p.add_argument("--seed-prefix", default="h2h")
args = p.parse_args()

def run(i):
    out = subprocess.run([sys.executable, ENGINE, args.a, args.b, "--hands", args.hands,
                          "--seed", f"{args.seed_prefix}{i}", "--log-dir", ""], capture_output=True, text=True).stdout
    net = int(re.search(r"A .*?net\s+([+-]\d+)", out).group(1))
    return net, "forfeit" in out

with cf.ThreadPoolExecutor(args.workers) as ex:
    res = list(ex.map(run, range(args.n)))
nets = [r[0] for r in res]
mean = statistics.mean(nets)
se = statistics.stdev(nets) / len(nets) ** 0.5 if len(nets) > 1 else float("nan")
if len(nets) < 20:
    verdict = "too few matches to judge - use 50 or more"
elif mean > 2 * se:
    verdict = "REAL improvement"
elif mean < -2 * se:
    verdict = "REAL regression"
else:
    verdict = "not distinguishable yet - run more matches or drop the change"
print(f"{os.path.basename(args.a)} vs {os.path.basename(args.b)}: wins {sum(n > 0 for n in nets)}/{args.n}, "
      f"mean {mean:+.1f} ± {se:.1f} chips/match ({mean/int(args.hands):+.3f}/hand)  -> {verdict}")
if any(r[1] for r in res):
    print("WARNING: a forfeit happened in at least one match")
