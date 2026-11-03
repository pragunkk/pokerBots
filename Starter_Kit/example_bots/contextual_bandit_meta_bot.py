"""
Radical Architecture 2: Online Multiplicative-Weights / Hedge Expert Portfolio Meta-Bot.
Instead of a hand-crafted classifier, this bot maintains an ensemble of 5 distinct specialist
strategies (Experts):
  0: GTO_MDF_EXPERT (Balanced Minimum Defense Frequency & polarized sizing)
  1: STATION_VALUE_EXPERT (Zero bluff, heavy thin value + geometric overbets)
  2: NIT_STEAL_EXPERT (100% button steals, relentless small-ball flop/turn/river probes)
  3: MANIAC_TRAP_EXPERT (Check-calls medium hands, snaps off river bluffs, traps preflop 3-bets)
  4: SMALL_BALL_META_EXPERT (Delayed turn floats, probe-sizing-tell punisher, tight river folds)
After every hand, the bot evaluates the counterfactual performance of each expert against the
observed opponent actions and updates expert weights via Exponential Multiplicative Weights (Hedge)!
"""
import os, sys, json, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import (
    RANK_VAL,
    board_danger_level,
    eval_cards,
    evaluate_discard_options,
    has_strong_draw,
    hole_interaction_level,
    post_discard_equity,
    preflop_3card_strength,
)


class HedgePortfolioBot:
    def __init__(self):
        # 5 Experts: [GTO, StationVal, NitSteal, ManiacTrap, SmallBallMeta]
        self.weights = [1.0, 0.8, 0.8, 0.8, 1.2]
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None
        self.seat = "button"
        self.active_expert = 4

    def select_expert_for_hand(self):
        # Softmax / Thompson-style selection heavily weighted toward the highest-weight expert
        best_i = max(range(5), key=lambda i: self.weights[i])
        if random.random() < 0.82:
            self.active_expert = best_i
        else:
            tot = sum(self.weights)
            r = random.random() * tot
            cum = 0.0
            for i, w in enumerate(self.weights):
                cum += w
                if r <= cum:
                    self.active_expert = i
                    break

    def update_expert_weights(self, msg):
        """Counterfactual Multiplicative Weights (Hedge) update from observed opponent behavior in the hand."""
        hist = msg.get("history", [])
        opp_acts = [h for h in hist if h["actor"] == "opp"]
        opp_raises = sum(1 for h in opp_acts if h["action"] == "raise")
        opp_calls = sum(1 for h in opp_acts if h["action"] == "call")
        opp_folds = sum(1 for h in opp_acts if h["action"] == "fold")

        rewards = [0.0, 0.0, 0.0, 0.0, 0.0]
        if opp_folds > 0 and opp_raises == 0:
            # Opponent folded to pressure -> NitSteal (2) and SmallBallMeta (4) excel!
            rewards[2] += 0.35
            rewards[4] += 0.20
        if opp_calls >= 2 and opp_raises == 0:
            # Opponent called multiple streets passively -> StationVal (1) excels, NitSteal (2) punished!
            rewards[1] += 0.45
            rewards[2] -= 0.35
        if opp_raises >= 2:
            # Opponent fired multiple raises/barrels -> ManiacTrap (3) excels!
            rewards[3] += 0.45
            rewards[2] -= 0.25
        if opp_raises == 1 and opp_calls <= 1:
            # Tight/balanced action -> SmallBallMeta (4) and GTO (0) excel!
            rewards[4] += 0.20
            rewards[0] += 0.15

        if msg.get("result") == "showdown" and msg.get("opp_hole_cards"):
            inter = hole_interaction_level(msg["opp_hole_cards"], msg["board"])
            if inter <= 1 and opp_raises >= 1:
                # Caught opponent bluffing at showdown!
                rewards[3] += 0.60
            elif inter >= 3 and opp_raises >= 1:
                # Opponent had nutted value when raising -> SmallBallMeta (4) / GTO (0) discipline rewarded
                rewards[4] += 0.35
                rewards[0] += 0.25
                rewards[3] -= 0.30

        eta = 0.25
        for i in range(5):
            self.weights[i] = max(0.10, min(50.0, self.weights[i] * math.exp(eta * rewards[i])))

    def choose_discard(self, req):
        best_c, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=75)
        return {"type": "discard", "card": best_c}

    def choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        ex = self.active_expert

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            open_t = [0.34, 0.38, 0.15, 0.36, 0.30][ex]
            if opp_bet <= 2:
                if s >= open_t and "raise" in legal:
                    sz = 7 if ex == 1 else (4 if ex == 2 else 5)
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], sz))}
                if to_call == 0 or s >= 0.22:
                    return {"type": "action", "action": "call" if to_call > 0 else "check"}
                return {"type": "action", "action": "fold"}
            else:
                if opp_bet >= 35:
                    req_s = 0.58 if ex == 3 else 0.76
                    return {"type": "action", "action": "call" if s >= req_s else "fold"}
                if s >= 0.68 and "raise" in legal and ex != 3:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], int(opp_bet * 2.8)))}
                call_t = 0.36 if ex == 3 else 0.44
                return {"type": "action", "action": "call" if s >= call_t else "fold"}

        if street == "flop":
            best_c, eq, _ = evaluate_discard_options(self.hole3, req["board"], num_rollouts=75)
            keep2 = [c for c in self.hole3 if c != best_c][:2]
            inter = hole_interaction_level(keep2, req["board"])
            has_draw = has_strong_draw(keep2, req["board"])
        else:
            bias = -0.20 if ex == 3 else (0.20 if to_call > 0 else 0.0)
            eq = post_discard_equity(self.hole2, req["board"], self.my_discard, self.opp_discard, opp_range_bias=bias, num_rollouts=85)
            inter = hole_interaction_level(self.hole2, req["board"])
            has_draw = has_strong_draw(self.hole2, req["board"])

        danger = board_danger_level(req["board"])
        pot_odds = to_call / max(1.0, pot + to_call)

        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}
            if ex == 3 and street in ("flop", "turn") and 0.48 <= eq <= 0.75:
                return {"type": "action", "action": "check"}
            if street == "river" and inter <= 1 and eq < 0.72:
                if ex in (2, 4) and inter == 0 and eq < 0.28 and random.random() < 0.45:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], int(pot * 0.38) + 1))}
                return {"type": "action", "action": "check"}
            if eq >= 0.56:
                frac = 0.80 if (ex == 1 or inter >= 3) else 0.52
                return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], int(pot * frac)))}
            if ex in (2, 4) and street != "river" and 0.36 <= eq < 0.56 and random.random() < (0.65 if ex == 2 else 0.38):
                return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], int(pot * 0.36)))}
            return {"type": "action", "action": "check"}
        else:
            if ex == 3:
                if inter >= 3 and eq >= 0.80 and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet + int(pot * 0.80)))}
                if street in ("turn", "river") and inter == 0 and eq < 0.58:
                    return {"type": "action", "action": "fold"}
                if inter >= 1 and eq >= max(0.42, pot_odds - 0.03):
                    return {"type": "action", "action": "call"}
            if street == "river" and inter == 0 and to_call > pot * 0.16:
                return {"type": "action", "action": "fold"}
            if street == "river" and inter == 1 and to_call > pot * 0.28:
                return {"type": "action", "action": "fold"}
            if street == "turn" and inter == 0 and not has_draw and to_call > pot * 0.25:
                return {"type": "action", "action": "fold"}
            if my_bet > 0 and opp_bet > my_bet and street in ("turn", "river") and inter <= 2:
                return {"type": "action", "action": "fold"}
            if inter >= 3 and eq >= 0.78 and danger < 0.45 and "raise" in legal:
                return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet + int((pot + to_call) * 0.75)))}
            if eq >= pot_odds + 0.02:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}


def main():
    bot = HedgePortfolioBot()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        t = msg.get("type")
        if t == "new_hand":
            bot.hole3 = list(msg["hole_cards"])
            bot.seat = msg["seat"]
            bot.my_discard = None
            bot.opp_discard = None
            bot.select_expert_for_hand()
        elif t == "discard_result":
            bot.my_discard = msg["my_discard"]
            bot.opp_discard = msg["opp_discard"]
            bot.hole2 = list(msg["hole_cards"])
        elif t == "hand_result":
            bot.update_expert_weights(msg)
        elif t == "action_request":
            print(json.dumps(bot.choose_action(msg)), flush=True)
        elif t == "discard_request":
            print(json.dumps(bot.choose_discard(msg)), flush=True)
        elif t == "match_end":
            break


if __name__ == "__main__":
    main()
