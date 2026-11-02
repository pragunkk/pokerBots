"""LAG / Maniac Bully Archetype: High pressure, wide 3-bets, frequent multi-street barrels."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import preflop_3card_strength, evaluate_discard_options, post_discard_equity

STATE = {"hole3": [], "hole2": [], "my_discard": None, "opp_discard": None}

def choose_action(req):
    legal = req["legal_actions"]
    street = req["street"]
    to_call = req["to_call"]
    pot = req["pot"]
    roll = random.random()

    if street == "preflop":
        eq = preflop_3card_strength(STATE["hole3"])
        if "raise" in legal and (eq > 0.35 or roll < 0.45):
            target = 8 if to_call <= 2 else req["opp_bet"] * 3
            return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], target))}
        if to_call == 0:
            return {"type": "action", "action": "check"}
        if eq > 0.32 or to_call <= 6:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

    if street == "flop":
        _, eq, _ = evaluate_discard_options(STATE["hole3"], req["board"], num_rollouts=40)
    else:
        eq = post_discard_equity(STATE["hole2"], req["board"], STATE["my_discard"], STATE["opp_discard"], num_rollouts=45)

    pot_odds = to_call / max(1, pot + to_call)
    if to_call == 0:
        if "raise" in legal and (eq > 0.48 or roll < 0.55):
            amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.85)))
            return {"type": "action", "action": "raise", "amount": amt}
        return {"type": "action", "action": "check"}
    else:
        if "raise" in legal and (eq > 0.65 or (eq > 0.40 and roll < 0.28)):
            amt = max(req["min_raise"], min(req["max_raise"], req["opp_bet"] + int(pot * 0.9)))
            return {"type": "action", "action": "raise", "amount": amt}
        if eq >= pot_odds - 0.03:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

def choose_discard(req):
    best_card, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=45)
    return {"type": "discard", "card": best_card}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        msg = json.loads(line)
        t = msg.get("type")
        if t == "new_hand":
            STATE["hole3"] = msg["hole_cards"]
            STATE["my_discard"] = None
            STATE["opp_discard"] = None
        elif t == "discard_result":
            STATE["my_discard"] = msg["my_discard"]
            STATE["opp_discard"] = msg["opp_discard"]
            STATE["hole2"] = msg["hole_cards"]
        elif t == "action_request":
            print(json.dumps(choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(choose_discard(msg)), flush=True)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
