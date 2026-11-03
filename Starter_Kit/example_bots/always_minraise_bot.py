"""Simple Heuristic 4: Min-Bet / Min-Raise Spammer — always raises min_raise to cap out the 4-raise limit cheaply."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import evaluate_discard_options

def choose_action(req):
    legal = req["legal_actions"]
    if "raise" in legal:
        return {"type": "action", "action": "raise", "amount": req["min_raise"]}
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
        if t == "action_request":
            print(json.dumps(choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(choose_discard(msg)), flush=True)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
