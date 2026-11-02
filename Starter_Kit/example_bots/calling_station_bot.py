"""Calling Station Archetype: Loose-Passive bot that calls wide and rarely bluffs."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import preflop_3card_strength, evaluate_discard_options, post_discard_equity

STATE = {"my_discard": None, "opp_discard": None}

def choose_action(req):
    legal = req["legal_actions"]
    street = req["street"]
    to_call = req["to_call"]
    pot = req["pot"]

    if street == "preflop":
        eq = preflop_3card_strength(req.get("hole_cards", ["2s", "7d", "9c"]))
    elif street == "flop":
        _, eq, _ = evaluate_discard_options(STATE["hole3"], req["board"], num_rollouts=35)
    else:
        eq = post_discard_equity(STATE["hole2"], req["board"], STATE["my_discard"], STATE["opp_discard"], num_rollouts=40)

    if to_call == 0:
        if eq > 0.72 and "raise" in legal:
            amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.5)))
            return {"type": "action", "action": "raise", "amount": amt}
        return {"type": "action", "action": "check"}

    # Calling station calls almost anything unless massive bet with terrible equity
    pot_odds = to_call / max(1, pot + to_call)
    if eq >= max(0.22, pot_odds - 0.15):
        if eq > 0.85 and "raise" in legal:
            amt = max(req["min_raise"], min(req["max_raise"], req["opp_bet"] + int(pot * 0.6)))
            return {"type": "action", "action": "raise", "amount": amt}
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold"}

def choose_discard(req):
    best_card, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=40)
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
            if msg["street"] == "preflop":
                STATE["hole3"] = STATE.get("hole3", ["As", "Kd", "2c"])
                msg["hole_cards"] = STATE["hole3"]
            print(json.dumps(choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(choose_discard(msg)), flush=True)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
