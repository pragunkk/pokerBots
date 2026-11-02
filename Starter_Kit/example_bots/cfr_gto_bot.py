"""
Game-Theory Optimal (GTO / MDF + Balanced Polarized Sizing) Rival Bot.
Represents a strong technical competitor using Minimum Defense Frequency (MDF),
balanced bluff-to-value ratios, and deceptive discards.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import (
    RANK_VAL,
    evaluate_discard_options,
    hole_interaction_level,
    post_discard_equity,
    preflop_3card_strength,
)

STATE = {"hole3": [], "hole2": [], "my_discard": None, "opp_discard": None, "seat": "button"}

def choose_discard(req):
    best_card, best_eq, eq_map = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=55)
    close = [c for c, e in eq_map.items() if best_eq - e <= 0.01]
    if len(close) > 1:
        best_card = random.choice(close)
    return {"type": "discard", "card": best_card}

def choose_action(req):
    legal = req["legal_actions"]
    street = req["street"]
    to_call = req["to_call"]
    pot = req["pot"]
    opp_bet = req["opp_bet"]

    if street == "preflop":
        s = preflop_3card_strength(STATE["hole3"])
        if opp_bet <= 2:
            if s >= 0.35 and "raise" in legal:
                return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], 6))}
            if to_call == 0 or s >= 0.24:
                return {"type": "action", "action": "call" if to_call > 0 else "check"}
            return {"type": "action", "action": "fold"}
        else:
            # Minimum Defense Frequency vs preflop raise
            if s >= 0.68 and "raise" in legal and opp_bet < 30:
                return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet * 3))}
            if s >= 0.44 and opp_bet <= 24:
                return {"type": "action", "action": "call"}
            if s >= 0.72:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

    if street == "flop":
        _, eq, _ = evaluate_discard_options(STATE["hole3"], req["board"], num_rollouts=55)
        inter = 2
    else:
        eq = post_discard_equity(STATE["hole2"], req["board"], STATE["my_discard"], STATE["opp_discard"],
                                 opp_range_bias=0.25 if to_call > 0 else 0.0, num_rollouts=65)
        inter = hole_interaction_level(STATE["hole2"], req["board"])

    pot_odds = to_call / max(1, pot + to_call)

    if to_call == 0:
        if "raise" in legal:
            # Balanced value + polarized bluff
            if eq >= 0.60 and inter >= 2:
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.66)))
                return {"type": "action", "action": "raise", "amount": amt}
            if street in ("flop", "turn") and 0.40 <= eq <= 0.48 and random.random() < 0.28:
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.66)))
                return {"type": "action", "action": "raise", "amount": amt}
        return {"type": "action", "action": "check"}
    else:
        if eq >= 0.76 and inter >= 2 and "raise" in legal:
            amt = max(req["min_raise"], min(req["max_raise"], opp_bet + int((pot + to_call) * 0.75)))
            return {"type": "action", "action": "raise", "amount": amt}
        if street == "river" and inter == 0 and to_call > 4:
            return {"type": "action", "action": "fold"}
        if eq >= pot_odds + 0.02:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold"}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        msg = json.loads(line)
        t = msg.get("type")
        if t == "new_hand":
            STATE["hole3"] = msg["hole_cards"]
            STATE["seat"] = msg["seat"]
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
