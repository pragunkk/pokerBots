#!/usr/bin/env python3
"""
Diverse General-Purpose Multi-Archetype Adversaries for RL Training (`rl_bot/`).
These synthetic adversaries are NOT tuned to any friend's bot — they represent
broad poker archetypes (Polarized Overbettor, Check-Raise Trapper, GTO-Mixed Caller,
and Trembling-Hand Hyper-LAG) so `NeuralCFRBot` generalizes against unseen tournament bots.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Starter_Kit"))
from fast_eval import (
    RANK_VAL,
    board_danger_level,
    eval_cards,
    evaluate_discard_options,
    hole_interaction_level,
    init_fast_tables,
    post_discard_equity,
    preflop_3card_strength,
)


class ParameterizedAdversary:
    def __init__(self, mode="gto_mixed"):
        self.mode = mode
        self.hole3 = []
        self.hole2 = []
        self.my_disc = None
        self.opp_disc = None
        self.rng = random.Random(hash(mode) & 0xFFFFFFFF)

    def handle_init(self, msg):
        init_fast_tables()

    def handle_new_hand(self, msg):
        self.hole3 = list(msg["hole_cards"])
        self.hole2 = []
        self.my_disc = None
        self.opp_disc = None

    def handle_discard_result(self, msg):
        self.my_disc = msg["my_discard"]
        self.opp_disc = msg["opp_discard"]
        self.hole2 = list(msg["hole_cards"])

    def choose_discard(self, req):
        best_c, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=90)
        return {"type": "discard", "card": best_c}

    def choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        opp_bet = req["opp_bet"]
        my_bet = req["my_bet"]

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            if self.mode == "hyper_lag":
                open_t, three_t, call_t = 0.24, 0.50, 0.30
            elif self.mode == "polarized_overbet":
                open_t, three_t, call_t = 0.34, 0.60, 0.36
            elif self.mode == "check_raise_trap":
                open_t, three_t, call_t = 0.36, 0.66, 0.38
            else:  # gto_mixed
                open_t, three_t, call_t = 0.30, 0.58, 0.34

            if opp_bet <= 2:
                if s >= open_t and "raise" in legal:
                    sz = 6 if (self.rng.random() < 0.5 or s >= 0.65) else 5
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], sz))}
                if to_call == 0:
                    return {"type": "action", "action": "check"}
                return {"type": "action", "action": "call" if s >= 0.18 else "fold"}
            if s >= three_t and "raise" in legal:
                amt = max(req["min_raise"], min(req["max_raise"], int(opp_bet * 2.7)))
                return {"type": "action", "action": "raise", "amount": amt}
            pot_odds = to_call / max(1.0, pot + to_call)
            if s >= max(call_t, pot_odds + 0.04) and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        # Postflop
        board = req["board"]
        if street == "flop":
            _, eq, _ = evaluate_discard_options(self.hole3, board, num_rollouts=90)
            keep2 = [c for c in self.hole3 if c != min(self.hole3, key=lambda c: RANK_VAL[c[0]])][:2]
            inter = hole_interaction_level(keep2, board)
        else:
            eq = post_discard_equity(self.hole2, board, self.my_disc, self.opp_disc, num_rollouts=120)
            inter = hole_interaction_level(self.hole2, board)

        pot_odds = to_call / max(1.0, pot + to_call)
        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}
            if self.mode == "check_raise_trap" and inter >= 3 and self.rng.random() < 0.65:
                return {"type": "action", "action": "check"}
            if self.mode == "polarized_overbet" and (eq >= 0.78 or (eq <= 0.25 and self.rng.random() < 0.30)):
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 1.25)))
                return {"type": "action", "action": "raise", "amount": amt}
            if self.mode == "hyper_lag" and (eq >= 0.54 or self.rng.random() < 0.35):
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.65)))
                return {"type": "action", "action": "raise", "amount": amt}
            if eq >= 0.60:
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.55)))
                return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": "check"}

        # Facing bet
        if self.mode == "check_raise_trap" and inter >= 3 and "raise" in legal:
            amt = max(req["min_raise"], min(req["max_raise"], opp_bet + int((pot + to_call) * 0.80)))
            return {"type": "action", "action": "raise", "amount": amt}
        if eq >= 0.82 and "raise" in legal:
            amt = max(req["min_raise"], min(req["max_raise"], opp_bet + int((pot + to_call) * 0.75)))
            return {"type": "action", "action": "raise", "amount": amt}
        if eq >= pot_odds + 0.03 and "call" in legal:
            return {"type": "action", "action": "call"}
        return {"type": "action", "action": "fold" if "fold" in legal else "check"}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gto_mixed"
    bot = ParameterizedAdversary(mode)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        t = msg.get("type")
        if t == "init":
            bot.handle_init(msg)
        elif t == "new_hand":
            bot.handle_new_hand(msg)
        elif t == "discard_result":
            bot.handle_discard_result(msg)
        elif t == "action_request":
            print(json.dumps(bot.choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(bot.choose_discard(msg)), flush=True)
        elif t == "match_end":
            break


if __name__ == "__main__":
    main()
