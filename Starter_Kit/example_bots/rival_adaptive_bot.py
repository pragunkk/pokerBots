"""
Rival Adaptive Gear-Switching Bot.
Simulates a dangerous hackathon competitor that plays Tight-Aggressive for the first 40 hands
to disguise its profile, then tracks opponent stats and switches gears dynamically!
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

class RivalBot:
    def __init__(self):
        self.hand_id = 0
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None
        self.opp_folds = 0
        self.opp_faced = 0
        self.opp_raises = 0
        self.opp_actions = 0

    def observe(self, history):
        for h in history:
            if h["actor"] == "opp":
                self.opp_actions += 1
                if h["action"] == "raise":
                    self.opp_raises += 1
                elif h["action"] == "fold":
                    self.opp_folds += 1
                    self.opp_faced += 1
                elif h["action"] == "call":
                    self.opp_faced += 1

    def choose_discard(self, req):
        best_c, best_eq, eq_map = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=50)
        return {"type": "discard", "card": best_c}

    def choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        opp_bet = req["opp_bet"]

        # Gear switching after hand 40!
        if self.hand_id <= 40:
            mode = "DISGUISE_TAG"
        else:
            fold_rate = self.opp_folds / max(1, self.opp_faced)
            agg_rate = self.opp_raises / max(1, self.opp_actions)
            if fold_rate > 0.48:
                mode = "EXPLOIT_BLUFF"
            elif agg_rate > 0.35:
                mode = "EXPLOIT_TRAP"
            else:
                mode = "VALUE_HEAVY"

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            open_t = 0.25 if mode == "EXPLOIT_BLUFF" else 0.42
            if opp_bet <= 2:
                if s >= open_t and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], 6))}
                if to_call == 0 or s >= 0.28:
                    return {"type": "action", "action": "call" if to_call > 0 else "check"}
                return {"type": "action", "action": "fold"}
            else:
                if s >= 0.66 and "raise" in legal and opp_bet <= 25:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet * 3))}
                if s >= (0.40 if mode == "EXPLOIT_TRAP" else 0.48) and opp_bet <= 30:
                    return {"type": "action", "action": "call"}
                if s >= 0.74:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

        if street == "flop":
            _, eq, _ = evaluate_discard_options(self.hole3, req["board"], num_rollouts=50)
        else:
            eq = post_discard_equity(self.hole2, req["board"], self.my_discard, self.opp_discard, num_rollouts=55)

        pot_odds = to_call / max(1, pot + to_call)
        if to_call == 0:
            if "raise" in legal:
                if eq >= 0.58 or (mode == "EXPLOIT_BLUFF" and random.random() < 0.45):
                    amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.70)))
                    return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": "check"}
        else:
            if eq >= 0.75 and "raise" in legal:
                amt = max(req["min_raise"], min(req["max_raise"], opp_bet + int(pot * 0.80)))
                return {"type": "action", "action": "raise", "amount": amt}
            margin = -0.04 if mode == "EXPLOIT_TRAP" else 0.03
            if eq >= pot_odds + margin:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

def main():
    bot = RivalBot()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        msg = json.loads(line)
        t = msg.get("type")
        if t == "new_hand":
            bot.hand_id = msg["hand_id"]
            bot.hole3 = msg["hole_cards"]
            bot.my_discard = None
            bot.opp_discard = None
        elif t == "discard_result":
            bot.my_discard = msg["my_discard"]
            bot.opp_discard = msg["opp_discard"]
            bot.hole2 = msg["hole_cards"]
        elif t == "hand_result":
            bot.observe(msg.get("history", []))
        elif t == "action_request":
            print(json.dumps(bot.choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(bot.choose_discard(msg)), flush=True)
        elif t == "match_end":
            break

if __name__ == "__main__":
    main()
