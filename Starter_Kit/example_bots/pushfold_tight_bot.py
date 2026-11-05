"""Sparring bot: preflop push-or-fold with a tight range (~18%: pairs 88+, AJ+); postflop check/fold."""
import sys, json
R = "23456789TJQKA"
def good(h):
    rs = sorted((R.index(c[0]) for c in h), reverse=True)
    if len(set(rs)) < 3 and max(r for r in rs if rs.count(r) > 1) >= 6: return True  # pairs 88+
    return rs[0] == 12 and rs[1] >= 9  # A + J or better
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    t = m.get("type")
    if t == "new_hand": hole = m["hole_cards"]
    elif t == "action_request":
        L = m["legal_actions"]
        if m["street"] == "preflop" and good(hole) and "raise" in L: r = {"action": "raise", "amount": m["max_raise"]}
        elif m["street"] == "preflop" and good(hole) and "call" in L: r = {"action": "call"}
        elif "check" in L: r = {"action": "check"}
        else: r = {"action": "fold"}
        r["type"] = "action"; print(json.dumps(r), flush=True)
    elif t == "discard_request":
        print(json.dumps({"type": "discard", "card": min(m["hole_cards"], key=lambda c: R.index(c[0]))}), flush=True)
    elif t == "match_end": break
