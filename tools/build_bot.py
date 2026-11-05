#!/usr/bin/env python3
"""Build the single-file submission Starter_Kit/bot.py from the three source modules.

    python tools/build_bot.py

Edit preflop_table.py, fast_eval.py and adaptive_bot.py, never bot.py by hand.
The build strips the imports between our own modules, joins the files in order,
then checks that the result compiles and imports only allowed modules."""
import ast, os, re, sys

KIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Starter_Kit")
SOURCES = ["preflop_table.py", "fast_eval.py", "adaptive_bot.py"]
LOCAL = {os.path.splitext(s)[0] for s in SOURCES}
ALLOWED = {"json", "random", "sys", "time", "math", "collections", "itertools", "functools",
           "bisect", "heapq", "array", "numpy", "scipy", "traceback"}

def strip_local_imports(text):
    # "from fast_eval import (\n  a,\n  b,\n)" and "from preflop_table import X"
    text = re.sub(r"^from (%s) import \([^)]*\)\n" % "|".join(LOCAL), "", text, flags=re.M)
    text = re.sub(r"^from (%s) import [^\n]*\n" % "|".join(LOCAL), "", text, flags=re.M)
    text = re.sub(r"^import (%s)\n" % "|".join(LOCAL), "", text, flags=re.M)
    return text

parts = ['"""GENERATED FILE - do not edit. Built by tools/build_bot.py from: %s"""\n' % ", ".join(SOURCES)]
for name in SOURCES:
    with open(os.path.join(KIT, name), encoding="utf-8") as f:
        parts.append(f"\n# {'=' * 20} {name} {'=' * 20}\n" + strip_local_imports(f.read()))
out = "".join(parts)

tree = ast.parse(out)  # raises SyntaxError if the bundle is broken
for node in ast.walk(tree):
    mods = []
    if isinstance(node, ast.Import):
        mods = [a.name.split(".")[0] for a in node.names]
    elif isinstance(node, ast.ImportFrom):
        mods = [(node.module or "").split(".")[0]]
    for mod in mods:
        if mod in LOCAL:
            sys.exit(f"build failed: local import of {mod} survived")
        if mod not in ALLOWED:
            sys.exit(f"build failed: import {mod} is not in the allowed list")

dest = os.path.join(KIT, "bot.py")
with open(dest, "w", encoding="utf-8") as f:
    f.write(out)
size = os.path.getsize(dest)
print(f"wrote {os.path.relpath(dest)}  {size/1024:.1f} KB  ({'OK' if size < 5 * 1024 * 1024 else 'TOO BIG'} for the 5 MB limit)")
