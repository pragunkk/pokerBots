"""Simple Heuristic 1: Always Fold Bot — folds whenever facing a bet, checks when free."""
import sys, json

RANKS = "23456789TJQKA"

def choose_action(req):
    legal = req["legal_actions"]
    if "check" in legal:
        return {"type": "action", "action": "check"}
    return {"type": "action", "action": "fold" if "fold" in legal else "call"}

def choose_discard(req):
    cards = req["hole_cards"]
    worst = min(cards, key=lambda c: RANKS.index(c[0]))
    return {"type": "discard", "card": worst}

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
