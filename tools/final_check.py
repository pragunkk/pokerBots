#!/usr/bin/env python3
"""One-command pre-submission check. Run from the repo root with Python 3.11:

    python3.11 tools/final_check.py

Builds bot.py, then checks: crash fuzzing, protocol edge cases, a chaos opponent through the real
engine, full Stage 1 and Stage 2 matches (notices, stderr, CPU time), stray prints, size, imports.
Prints PASS/FAIL per check and exits non-zero if anything fails."""
import os, re, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(ROOT, "Starter_Kit")
BOT = os.path.join(KIT, "bot.py")
PY = sys.executable
results = []

def run(cmd, timeout=900):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT)

def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""), flush=True)

if sys.version_info[:2] != (3, 11):
    print(f"WARNING: running Python {sys.version.split()[0]}; the judges use 3.11. Use python3.11.")

r = run([PY, "tools/build_bot.py"])
check("build bot.py from sources", r.returncode == 0, r.stdout.strip() or r.stderr.strip()[-200:])

src = open(BOT, encoding="utf-8").read()
size_kb = os.path.getsize(BOT) / 1024
check("size under 5 MB", size_kb < 5 * 1024, f"{size_kb:.1f} KB")
prints = [i + 1 for i, line in enumerate(src.splitlines()) if re.search(r"\bprint\(", line) and not line.strip().startswith("#")]
check("no print() calls (stdout is protocol only)", not prints, f"lines {prints[:5]}" if prints else "")
check("no file access (open/write)", not re.search(r"\bopen\(", src))

r = run([PY, "tools/fuzz.py", BOT, "10000"])
check("fuzz 10,000 random states", r.returncode == 0, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:])

edge = os.path.join(ROOT, "tools", "edge_test.py")
if os.path.exists(edge):
    r = run([PY, edge, BOT])
    check("protocol edge cases", "ALL EDGE CASES PASSED" in r.stdout, r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-200:])

def match(opp, stage, seed):
    logdir = os.path.join(ROOT, "logs", f"final_{os.path.basename(opp)[:-3]}_{stage}")
    t = time.time()
    r = run([PY, os.path.join(KIT, "engine.py"), BOT, opp, "--stage", str(stage), "--seed", seed, "--log-dir", logdir])
    m = re.search(r"A .*?net\s+([+-]\d+).*?notices (\d+)", r.stdout)
    err = 0
    for leg in (1, 2):
        p = os.path.join(logdir, f"A_leg{leg}_stderr.txt")
        if os.path.exists(p):
            err += os.path.getsize(p)
    forfeit = "forfeit" in r.stdout.lower()
    ok = bool(m) and m.group(2) == "0" and err == 0 and not forfeit
    detail = f"net {m.group(1)}, notices {m.group(2)}, stderr {err} B, {time.time() - t:.0f}s wall" if m else r.stdout[-300:]
    return ok, detail

chaos = os.path.join(KIT, "example_bots", "chaos_bot.py")
if os.path.exists(chaos):
    ok, d = match(chaos, 2, "final_chaos")
    check("Stage 2 vs chaos bot (junk + illegal moves)", ok, d)
champs = sorted(f for f in os.listdir(os.path.join(ROOT, "champions")) if f.endswith(".py"))
opp = os.path.join(ROOT, "champions", champs[-1]) if champs else os.path.join(KIT, "example_bots", "tag_bot.py")
ok, d = match(opp, 1, "final_s1")
check(f"Stage 1 match vs {os.path.basename(opp)}", ok, d)
ok, d = match(opp, 2, "final_s2")
check(f"Stage 2 match vs {os.path.basename(opp)}", ok, d)

print("\nALL CHECKS PASSED - safe to upload Starter_Kit/bot.py" if all(results) else "\nSOMETHING FAILED - do not upload, fix it first")
sys.exit(0 if all(results) else 1)
