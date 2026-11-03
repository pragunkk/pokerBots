"""Simple Heuristic 5: Always Pot-Bet Bot — always bets/raises 100% pot on every street."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import evaluate_discard_options

def choose_action(req):
    legal = req["legal_actions"]
    if "raise" in legal:
        target = req["opp_bet"] + req["pot"] + req["to_call"]
        amt = max(req["min_raise"], min(req["max_raise"], target))
        return {"type": "action", "action": "raise", "amount": amt}
    if "call" in legal:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "check"}

def choose_discard(req):
    best_c, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=35)
    return {"type": "discard", "card": best_c}

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        msg = json.loads(line)
        t = msg.get("type")
        if t == "new_hand":
            pass
        elif t == "action_request":
            print(json.dumps(choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(choose_discard(msg)), flush=True)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
