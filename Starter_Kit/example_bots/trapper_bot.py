"""Trapper / Deceptive Archetype: Slowplays monsters on flop/turn to check-raise, uses deceptive discards."""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import preflop_3card_strength, evaluate_discard_options, post_discard_equity, RANK_VAL

STATE = {"hole3": [], "hole2": [], "my_discard": None, "opp_discard": None}

def choose_action(req):
    legal = req["legal_actions"]
    street = req["street"]
    to_call = req["to_call"]
    pot = req["pot"]

    if street == "preflop":
        eq = preflop_3card_strength(STATE["hole3"])
        if to_call <= 2:
            # Slowplay very strong hands sometimes, raise medium-strong
            if 0.48 <= eq <= 0.78 and "raise" in legal:
                return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], 6))}
            if eq > 0.32:
                return {"type": "action", "action": "call" if to_call > 0 else "check"}
            return {"type": "action", "action": "check" if "check" in legal else "fold"}
        else:
            if eq >= 0.45:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

    if street == "flop":
        _, eq, _ = evaluate_discard_options(STATE["hole3"], req["board"], num_rollouts=45)
    else:
        eq = post_discard_equity(STATE["hole2"], req["board"], STATE["my_discard"], STATE["opp_discard"], num_rollouts=50)

    pot_odds = to_call / max(1, pot + to_call)
    if to_call == 0:
        # Check-trap with monsters on flop/turn, bomb river
        if street in ("flop", "turn") and eq > 0.80 and random.random() < 0.65:
            return {"type": "action", "action": "check"}
        if eq > 0.62 and "raise" in legal:
            amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.70)))
            return {"type": "action", "action": "raise", "amount": amt}
        return {"type": "action", "action": "check"}
    else:
        # Check-raise springs the trap!
        if eq > 0.75 and "raise" in legal:
            amt = max(req["min_raise"], min(req["max_raise"], req["opp_bet"] + int(pot * 1.1)))
            return {"type": "action", "action": "raise", "amount": amt}
        if eq >= pot_odds:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

def choose_discard(req):
    best_card, best_eq, eq_map = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=50)
    # Deceptive discard: if two discards are within 2% equity, discard the HIGHER rank card to fake monster
    candidates = [c for c, e in eq_map.items() if best_eq - e <= 0.02]
    chosen = max(candidates, key=lambda c: RANK_VAL[c[0]])
    return {"type": "discard", "card": chosen}

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
