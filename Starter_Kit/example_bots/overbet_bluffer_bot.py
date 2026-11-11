"""Sparring bot: calls preflop, then bets 1.5x pot on the turn and river with ANY hand (pure overbet bluffer)."""
import sys, json
R = "23456789TJQKA"
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    t = m.get("type")
    if t == "action_request":
        L = m["legal_actions"]; st = m["street"]
        if st in ("turn", "river") and "raise" in L and m["my_bet"] == 0:
            amt = m["opp_bet"] + int((m["pot"] + m["to_call"]) * 1.5)
            r = {"action": "raise", "amount": max(m["min_raise"], min(m["max_raise"], amt))}
        elif "check" in L: r = {"action": "check"}
        elif m["to_call"] <= 12 or st in ("turn", "river"): r = {"action": "call"}
        else: r = {"action": "fold"}
        r["type"] = "action"; print(json.dumps(r), flush=True)
    elif t == "discard_request":
        print(json.dumps({"type": "discard", "card": min(m["hole_cards"], key=lambda c: R.index(c[0]))}), flush=True)
    elif t == "match_end": break
