"""Adversarial sparring bot. MODE selects the exploit attempt."""
MODE = "smallbet30"
import sys, json, random
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fast_eval as fe
fe.init_fast_tables()

def made(hole, board):
    """Best interaction level using any 2 of our cards (0 air, 1 pair, 2 top pair-ish, 3 two pair+)."""
    if len(hole) == 2: return fe.hole_interaction_level(hole, board)
    return max(fe.hole_interaction_level([hole[i], hole[j]], board) for i, j in ((0,1),(0,2),(1,2)))
def pre(hole): return fe.preflop_3card_strength(hole)
def top(x): return fe.strength_quantile(1.0 - x)
def R(m, amt): return {"action": "raise", "amount": max(m["min_raise"], min(m["max_raise"], int(amt)))}
def potfrac(m, f): return R(m, m["opp_bet"] + (m["pot"] + m["to_call"]) * f)
CALL, CHECK, FOLD = {"action": "call"}, {"action": "check"}, {"action": "fold"}
def passive(m, ok):
    L = m["legal_actions"]
    if "check" in L: return CHECK
    return CALL if ok else FOLD

def decide(m, hole):
    L = m["legal_actions"]; st = m["street"]; tc = m["to_call"]; pot = m["pot"]; can = "raise" in L
    s = pre(hole) if st == "preflop" else None
    lvl = made(hole, m["board"]) if st != "preflop" else 0
    ratio = tc / max(1, pot - tc)
    if MODE == "threebet_spam":
        if st == "preflop":
            if can and m["raises_left"] >= 2: return R(m, max(6, m["opp_bet"] * 3))
            return passive(m, s >= top(0.5) or tc <= 6)
        if st == "flop" and can and tc == 0: return potfrac(m, 0.6)
        return passive(m, lvl >= 1 and ratio <= 1.0 or lvl >= 3)
    if MODE in ("smallbet30", "smallbet45", "smallbet30_raisey"):
        f = 0.45 if MODE == "smallbet45" else 0.30
        if st == "preflop": return passive(m, tc <= 6 or s >= top(0.2))
        if can and tc == 0: return potfrac(m, f)
        if MODE == "smallbet30_raisey" and can and tc > 0 and lvl >= 3: return potfrac(m, 0.8)
        return passive(m, (lvl >= 1 and ratio <= 0.6) or lvl >= 3)
    if MODE == "minbet":
        if st == "preflop": return passive(m, tc <= 6 or s >= top(0.2))
        if can and tc == 0: return R(m, m["min_raise"])
        return passive(m, (lvl >= 1 and ratio <= 0.6) or lvl >= 3)
    if MODE == "checkraise":
        if st == "preflop": return passive(m, tc <= 8 or s >= top(0.15))
        if tc > 0 and can and m["my_bet"] == 0 and st in ("flop", "turn"): return R(m, m["opp_bet"] * 3)
        if tc == 0: return CHECK
        return passive(m, lvl >= 2 or (lvl >= 1 and ratio <= 0.5))
    if MODE == "limp_shove":
        if st == "preflop":
            if tc > 1 and can and s >= top(0.35): return R(m, m["max_raise"])
            if tc > 1: return passive(m, s >= top(0.35) or tc <= 2)
            return CALL if "call" in L else CHECK
        if can and tc == 0 and lvl >= 2: return potfrac(m, 0.7)
        return passive(m, lvl >= 2 or (lvl >= 1 and ratio <= 0.5))
    if MODE == "barrel":
        if st == "preflop": return passive(m, tc <= 8 or s >= top(0.15))
        if can and tc == 0: return potfrac(m, 0.7)
        return passive(m, lvl >= 3 or (lvl >= 2 and ratio <= 0.8))
    if MODE in ("pot_honest", "pot105_honest"):
        if st == "preflop":
            if can and s >= top(0.25) and m["raises_left"] >= 2: return R(m, max(6, m["opp_bet"] * 3))
            return passive(m, tc <= 8 or s >= top(0.2))
        f = 1.0 if MODE == "pot_honest" else 1.08
        if can and lvl >= 3 and st in ("turn", "river"): return potfrac(m, f)
        if can and tc == 0 and lvl >= 2 and st == "flop": return potfrac(m, 0.5)
        return passive(m, lvl >= 2 and ratio <= 0.7 or lvl >= 3)
    if MODE == "trap_river_shove":
        if st == "preflop": return passive(m, tc <= 8 or s >= top(0.2))
        if st == "river" and can and lvl >= 3: return R(m, m["max_raise"])
        return passive(m, lvl >= 2 or (lvl >= 1 and ratio <= 0.6))
    if MODE == "polar_overbet":
        if st == "preflop": return passive(m, tc <= 8 or s >= top(0.2))
        if st == "river" and can and tc == 0 and (lvl >= 3 or (lvl == 0 and random.random() < 0.6)): return potfrac(m, 1.25)
        if can and tc == 0 and lvl >= 2: return potfrac(m, 0.6)
        return passive(m, lvl >= 2 or (lvl >= 1 and ratio <= 0.6))
    if MODE == "open_big":
        if st == "preflop":
            if can and tc <= 1 and s >= top(0.6): return R(m, 14)
            if can and s >= top(0.08): return R(m, m["max_raise"])
            return passive(m, s >= top(0.3) and tc <= 14)
        if can and tc == 0 and (lvl >= 1 or random.random() < 0.4): return potfrac(m, 0.66)
        return passive(m, lvl >= 2 or (lvl >= 1 and ratio <= 0.7))
    if MODE == "fourbet_jam":
        if st == "preflop":
            if can and tc <= 1: return R(m, 5)
            if can and m["opp_bet"] >= 10: return R(m, m["max_raise"]) if s >= top(0.45) else FOLD
            return passive(m, tc <= 6 or s >= top(0.3))
        if can and tc == 0 and (lvl >= 1 or random.random() < 0.3): return potfrac(m, 0.6)
        return passive(m, lvl >= 2 or (lvl >= 1 and ratio <= 0.7))
    if MODE == "station_raiser":   # calls everything, raises big with two pair+
        if st == "preflop": return passive(m, True)
        if can and lvl >= 3: return potfrac(m, 1.0)
        return passive(m, lvl >= 1 or ratio <= 0.35)
    return passive(m, False)

hole = []
for line in sys.stdin:
    try: m = json.loads(line)
    except ValueError: continue
    t = m.get("type")
    if t == "new_hand": hole = m["hole_cards"]
    elif t == "discard_result": hole = m["hole_cards"]
    elif t == "discard_request":
        h = m["hole_cards"]; b = m["board"]
        best = max(h, key=lambda d: (fe.eval_cards([c for c in h if c != d], b), random.random()))
        print(json.dumps({"type": "discard", "card": best}), flush=True)
    elif t == "action_request":
        try: r = dict(decide(m, hole))
        except Exception: r = dict(passive(m, False))
        if r["action"] not in m["legal_actions"]: r = dict(passive(m, r["action"] in ("raise", "call")))
        r["type"] = "action"; print(json.dumps(r), flush=True)
    elif t == "match_end": break
