#!/usr/bin/env python3
"""Play one bot against every bot in Starter_Kit/example_bots over several seeds.

    python tools/gauntlet.py Starter_Kit/bot.py --seeds 6 [--workers 4] [--stage 1]"""
import argparse, glob, os, re, statistics, subprocess, sys, concurrent.futures as cf

KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Starter_Kit")
p = argparse.ArgumentParser()
p.add_argument("bot"); p.add_argument("--seeds", type=int, default=6)
p.add_argument("--workers", type=int, default=os.cpu_count() or 2); p.add_argument("--stage", default="1")
args = p.parse_args()
opps = sorted(glob.glob(os.path.join(KIT, "example_bots", "*.py")))

def run(job):
    opp, seed = job
    out = subprocess.run([sys.executable, os.path.join(KIT, "engine.py"), args.bot, opp, "--stage", args.stage,
                          "--seed", f"g{seed}", "--log-dir", ""], capture_output=True, text=True).stdout
    m = re.search(r"A .*?net\s+([+-]\d+)\s+time used\s+([\d.]+) s.*?notices (\d+)", out)
    return opp, int(m.group(1)), float(m.group(2)), int(m.group(3)), "forfeit" in out

res = {}
with cf.ThreadPoolExecutor(args.workers) as ex:
    for r in ex.map(run, [(o, s) for o in opps for s in range(args.seeds)]):
        res.setdefault(r[0], []).append(r)
W = L = 0
print(f"{'opponent':30s} {'W-L':>6s} {'mean net':>9s} {'worst':>7s} {'max time s':>10s} notices forfeit")
for o in opps:
    rs = res[o]; nets = [r[1] for r in rs]
    w, l = sum(n > 0 for n in nets), sum(n < 0 for n in nets); W += w; L += l
    print(f"{os.path.basename(o):30s} {w:>2d}-{l:<3d} {statistics.mean(nets):9.1f} {min(nets):7d} "
          f"{max(r[2] for r in rs):10.2f} {sum(r[3] for r in rs):7d} {'YES' if any(r[4] for r in rs) else 'no'}")
print(f"TOTAL {W}-{L}")
