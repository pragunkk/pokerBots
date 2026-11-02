"""
Proven Algorithm 5: Tartanian/Libratus Pseudoharmonic Action Translation + Blocker-Bluffing Bot.
Implements:
  - Ganzfried & Sandholm Pseudoharmonic Mapping f(x) = (x - A)*(1 + B) / ((B - A)*(1 + x))
    to translate arbitrary opponent bet sizes into canonical game-tree probabilities
  - Dead-Card & Discard Blocker Bluffing: on scary boards (3-flush or 4-straight), checks if
    our hole cards + `my_discard` hold the Ace/King blocker of the flush suit and fires
    high-leverage blocker bluffs!
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import (
    RANK_VAL,
    SUIT_VAL,
    evaluate_discard_options,
    hole_interaction_level,
    post_discard_equity,
    preflop_3card_strength,
)


def pseudoharmonic_prob(x, a, b):
    """Maps an off-tree bet size x (as pot fraction in [a, b]) to probability of selecting upper abstraction b."""
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    return ((x - a) * (1.0 + b)) / ((b - a) * (1.0 + x))


class SlumbotBlockerHybrid:
    def __init__(self):
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None

    def holds_nut_flush_blocker(self, board_strs):
        suits = [c[1] for c in board_strs]
        for s in "shdc":
            if suits.count(s) >= 3:
                # 3+ cards of suit s on board: check if we hold As or Ks in hole2 (or discarded it)
                my_known = list(self.hole2) + ([self.my_discard] if self.my_discard else [])
                if ("A" + s) in my_known or ("K" + s) in my_known:
                    # Make sure we don't actually have the flush (2 cards of suit s in hole2)
                    if sum(1 for c in self.hole2 if c[1] == s) < 2:
                        return True
        return False

    def choose_discard(self, req):
        best_c, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=65)
        return {"type": "discard", "card": best_c}

    def choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        opp_bet = req["opp_bet"]

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            if opp_bet <= 2:
                if s >= 0.34 and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], 6))}
                if to_call == 0 or s >= 0.22:
                    return {"type": "action", "action": "call" if to_call > 0 else "check"}
                return {"type": "action", "action": "fold"}
            else:
                if s >= 0.67 and "raise" in legal and opp_bet <= 25:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet * 3))}
                if s >= 0.43 and opp_bet <= 30:
                    return {"type": "action", "action": "call"}
                if s >= 0.74:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

        if street == "flop":
            _, eq, _ = evaluate_discard_options(self.hole3, req["board"], num_rollouts=65)
            inter = 2
        else:
            eq = post_discard_equity(
                self.hole2, req["board"], self.my_discard, self.opp_discard,
                opp_range_bias=0.20 if to_call > 0 else 0.0, num_rollouts=75
            )
            inter = hole_interaction_level(self.hole2, req["board"])

        pot_odds = to_call / max(1, pot + to_call)

        if to_call == 0:
            if "raise" in legal:
                if eq >= 0.58 and (street != "river" or inter >= 2):
                    amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.65)))
                    return {"type": "action", "action": "raise", "amount": amt}
                # Nut-Flush Blocker Bluff on Turn/River!
                if street in ("turn", "river") and eq < 0.45 and self.holds_nut_flush_blocker(req["board"]):
                    if random.random() < 0.55:
                        amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.72)))
                        return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": "check"}
        else:
            # Translate opponent bet size using Pseudoharmonic Mapping between 0.33P (small) and 1.0P (polar)
            bet_frac = to_call / max(1.0, pot - to_call)
            p_polar = pseudoharmonic_prob(bet_frac, 0.33, 1.0)
            req_eq = pot_odds + 0.01 + 0.05 * p_polar

            if street == "river" and inter == 0 and to_call > pot * 0.18:
                return {"type": "action", "action": "fold"}
            if eq >= 0.78 and inter >= 3 and "raise" in legal:
                amt = max(req["min_raise"], min(req["max_raise"], opp_bet + int((pot + to_call) * 0.75)))
                return {"type": "action", "action": "raise", "amount": amt}
            if eq >= req_eq:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}


def main():
    bot = SlumbotBlockerHybrid()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        t = msg.get("type")
        if t == "new_hand":
            bot.hole3 = msg["hole_cards"]
            bot.my_discard = None
            bot.opp_discard = None
        elif t == "discard_result":
            bot.my_discard = msg["my_discard"]
            bot.opp_discard = msg["opp_discard"]
            bot.hole2 = msg["hole_cards"]
        elif t == "action_request":
            print(json.dumps(bot.choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(bot.choose_discard(msg)), flush=True)
        elif t == "match_end":
            break


if __name__ == "__main__":
    main()
