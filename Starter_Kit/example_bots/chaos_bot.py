"""Sparring bot for robustness: random legal actions with extreme sizes (min-raise wars to the cap,
shoves on every street), occasional ILLEGAL replies and junk lines, random discards."""
import sys, json, random
R = "23456789TJQKA"
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    t = m.get("type")
    if t == "action_request":
        L = m["legal_actions"]; x = random.random()
        if x < 0.03:
            print("garbage line", flush=True); continue            # malformed -> engine corrects
        if x < 0.06:
            print(json.dumps({"type": "action", "action": "raise", "amount": -5}), flush=True); continue  # illegal
        if "raise" in L and x < 0.55:
            amt = random.choice([m["min_raise"], m["max_raise"], (m["min_raise"] + m["max_raise"]) // 2])
            r = {"action": "raise", "amount": amt}
        elif "call" in L and x < 0.85: r = {"action": "call"}
        elif "check" in L: r = {"action": "check"}
        else: r = {"action": random.choice(L)}
        r["type"] = "action"; print(json.dumps(r), flush=True)
    elif t == "discard_request":
        c = random.choice(m["hole_cards"] + ["Zz"])                   # sometimes illegal
        print(json.dumps({"type": "discard", "card": c}), flush=True)
    elif t == "match_end": break
