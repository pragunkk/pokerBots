"""Sparring bot: plain Monte-Carlo equity vs random hands, pot-odds calls, 2/3-pot value bets, some bluffs.
Represents a 'sound but unexploitative' competitor."""
import sys, json, random, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Starter_Kit/
import fast_eval as fe
D = list(fe.ALL_CARDS)
def eq(hole, board, dead, n=150):
    rest = [c for c in D if c not in hole and c not in board and c not in dead]
    w = 0.0
    k3 = len(hole) == 3
    for _ in range(n):
        s = random.sample(rest, 2 + (5 - len(board)))
        b = board + s[2:]
        if k3:
            mine = max(fe.eval_cards([hole[i], hole[j]], b) for i, j in ((0,1),(0,2),(1,2)))
        else:
            mine = fe.eval_cards(hole, b)
        o = fe.eval_cards(s[:2], b)
        w += 1.0 if mine > o else 0.5 if mine == o else 0.0
    return w / n
hole = []; dead = []
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    t = m.get("type")
    if t == "new_hand": hole = m["hole_cards"]; dead = []
    elif t == "discard_result": hole = m["hole_cards"]; dead = [m["my_discard"], m["opp_discard"]]
    elif t == "discard_request":
        h = m["hole_cards"]; b = m["board"]
        best = max(h, key=lambda d: eq([c for c in h if c != d], b, [d], 200))
        print(json.dumps({"type": "discard", "card": best}), flush=True)
    elif t == "action_request":
        L = m["legal_actions"]; e = eq(hole, m["board"], dead, 120 if m["board"] else 80)
        if not m["board"]: e -= 0.08  # 3-card preflop equity vs 2 random cards overstates strength
        pot, tc = m["pot"], m["to_call"]
        def rz(f):
            return {"action": "raise", "amount": max(m["min_raise"], min(m["max_raise"], m["opp_bet"] + int((pot + tc) * f)))}
        if e > 0.72 and "raise" in L: r = rz(0.66)
        elif tc == 0 and "raise" in L and (e > 0.60 or random.random() < 0.12): r = rz(0.5)
        elif tc > 0 and e >= tc / (pot + tc) + 0.03: r = {"action": "call"}
        elif "check" in L: r = {"action": "check"}
        else: r = {"action": "fold"}
        r["type"] = "action"; print(json.dumps(r), flush=True)
    elif t == "match_end": break
