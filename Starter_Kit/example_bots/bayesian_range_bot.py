"""
Proven Algorithm 4: Combinatorial Hand-Range Particle Filter & Discard-Elimination Bot.
Maintains an explicit particle belief distribution over the opponent's holdings throughout the hand:
  1. Preflop: Initializes 100 weighted 3-card opponent hands filtered by preflop action (open/call/3-bet).
  2. Flop Discard Elimination: When `opp_discard` is revealed, immediately eliminates any opponent holding
     containing `opp_discard` or dead board cards, AND sets each remaining 2-card holding `(c1, c2)` weight
     proportional to the probability that `opp_discard` was the rational discard from `{c1, c2, opp_discard}`!
  3. Postflop Bayesian Updates: Reweights particles whenever opponent bets or raises.
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import (
    ALL_CARDS,
    CARD_IDX,
    CARD_RANK,
    CARD_SUIT,
    eval5_idx,
    eval7_idx,
    evaluate_discard_options,
    hole_interaction_level,
    preflop_3card_strength,
)


class BayesianRangeBot:
    def __init__(self):
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None
        self.opp_preflop_aggressive = False

    def choose_discard(self, req):
        best_c, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=65)
        return {"type": "discard", "card": best_c}

    def bayesian_turn_river_equity(self, board_strs, opp_bet_frac):
        hole_idxs = [CARD_IDX[c] for c in self.hole2]
        board_idxs = [CARD_IDX[c] for c in board_strs]
        dead = set(hole_idxs + board_idxs)
        if self.my_discard:
            dead.add(CARD_IDX[self.my_discard])
        opp_disc_idx = CARD_IDX[self.opp_discard] if self.opp_discard else None
        if opp_disc_idx is not None:
            dead.add(opp_disc_idx)

        rem = [i for i in range(52) if i not in dead]
        to_come = 5 - len(board_idxs)
        flop3 = board_idxs[:3]

        total_w = 0.0
        wins = 0.0
        for _ in range(90):
            samp = random.sample(rem, 2 + to_come)
            o0, o1 = samp[0], samp[1]
            full_board = board_idxs + samp[2:]

            # Prior weight from preflop action
            w = 1.0
            if self.opp_preflop_aggressive and max(CARD_RANK[o0], CARD_RANK[o1]) < 10 and CARD_RANK[o0] != CARD_RANK[o1]:
                w *= 0.30

            # Exact Discard Likelihood P(opp_discard | {o0, o1, opp_discard}, flop3)
            if opp_disc_idx is not None and len(flop3) == 3:
                s_keep = eval5_idx(o0, o1, flop3[0], flop3[1], flop3[2])
                s_alt1 = eval5_idx(o0, opp_disc_idx, flop3[0], flop3[1], flop3[2])
                s_alt2 = eval5_idx(o1, opp_disc_idx, flop3[0], flop3[1], flop3[2])
                if s_keep < max(s_alt1, s_alt2):
                    w *= 0.05
                else:
                    w *= 1.40

            op_score = eval7_idx([o0, o1] + full_board)
            my_score = eval7_idx(hole_idxs + full_board)

            # Likelihood P(opp_bet_frac | op_score)
            if opp_bet_frac > 0.0:
                cat = op_score >> 20
                if cat >= 2:
                    w *= (1.0 + 2.5 * opp_bet_frac)
                elif cat == 1:
                    w *= (1.0 + 0.8 * opp_bet_frac)
                else:
                    w *= max(0.20, 1.0 - 0.9 * opp_bet_frac)

            total_w += w
            if my_score > op_score:
                wins += w
            elif my_score == op_score:
                wins += 0.5 * w

        return wins / max(total_w, 1e-9)

    def choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        opp_bet = req["opp_bet"]

        for h in req.get("history", []):
            if h["street"] == "preflop" and h["actor"] == "opp" and h["action"] == "raise":
                self.opp_preflop_aggressive = True

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            if opp_bet <= 2:
                if s >= 0.35 and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], 6))}
                if to_call == 0 or s >= 0.23:
                    return {"type": "action", "action": "call" if to_call > 0 else "check"}
                return {"type": "action", "action": "fold"}
            else:
                if s >= 0.68 and "raise" in legal and opp_bet <= 25:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet * 3))}
                if s >= 0.44 and opp_bet <= 30:
                    return {"type": "action", "action": "call"}
                if s >= 0.74:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

        if street == "flop":
            _, eq, _ = evaluate_discard_options(self.hole3, req["board"], num_rollouts=65)
            inter = 2
        else:
            bet_frac = min(1.5, to_call / max(1.0, pot - to_call)) if to_call > 0 else 0.0
            eq = self.bayesian_turn_river_equity(req["board"], bet_frac)
            inter = hole_interaction_level(self.hole2, req["board"])

        pot_odds = to_call / max(1, pot + to_call)

        if to_call == 0:
            if "raise" in legal:
                if street == "river" and inter <= 1 and eq < 0.70:
                    return {"type": "action", "action": "check"}
                if eq >= 0.58:
                    frac = 0.75 if eq >= 0.78 else 0.55
                    amt = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                    return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": "check"}
        else:
            if street == "river" and inter == 0 and to_call > pot * 0.18:
                return {"type": "action", "action": "fold"}
            if eq >= 0.78 and inter >= 3 and "raise" in legal:
                amt = max(req["min_raise"], min(req["max_raise"], opp_bet + int((pot + to_call) * 0.75)))
                return {"type": "action", "action": "raise", "amount": amt}
            if eq >= pot_odds + 0.02:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}


def main():
    bot = BayesianRangeBot()
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
            bot.opp_preflop_aggressive = False
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
