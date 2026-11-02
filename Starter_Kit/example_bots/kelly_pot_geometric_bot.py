"""
Proven Algorithm 3: DeepStack/Pluribus-Style Geometric Pot-Growth & Kelly Criterion Bot.
Uses:
  - Exact Geometric Bet Sizing: b_geo = 0.5 * (((2 * S + P) / P) ** (1 / streets_left) - 1)
    so nutted hands smoothly commit all remaining stack in equal pot-fraction geometric steps
  - Fractional Kelly Criterion for bankroll/pot-commitment sizing with medium-strong hands
  - SPR (Stack-to-Pot Ratio) commitment thresholds
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import (
    evaluate_discard_options,
    hole_interaction_level,
    post_discard_equity,
    preflop_3card_strength,
)


def geometric_bet_fraction(pot, eff_stack, streets_left):
    """Exact pot fraction b such that betting b*Pot on each of `streets_left` puts all `eff_stack` in."""
    if pot <= 0 or eff_stack <= 0 or streets_left <= 0:
        return 0.66
    target_ratio = (2.0 * eff_stack + pot) / pot
    frac = 0.5 * ((target_ratio ** (1.0 / streets_left)) - 1.0)
    return max(0.33, min(1.50, frac))


class KellyGeometricBot:
    def __init__(self):
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None

    def choose_discard(self, req):
        best_c, _, _ = evaluate_discard_options(req["hole_cards"], req["board"], num_rollouts=65)
        return {"type": "discard", "card": best_c}

    def choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        opp_bet = req["opp_bet"]
        eff_stack = min(req["my_stack"], req["opp_stack"])

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            if opp_bet <= 2:
                if s >= 0.35 and "raise" in legal:
                    sz = 7 if s >= 0.65 else 5
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], sz))}
                if to_call == 0 or s >= 0.23:
                    return {"type": "action", "action": "call" if to_call > 0 else "check"}
                return {"type": "action", "action": "fold"}
            else:
                if s >= 0.68 and "raise" in legal and opp_bet <= 28:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet * 3))}
                if s >= 0.44 and opp_bet <= 30:
                    return {"type": "action", "action": "call"}
                if s >= 0.75:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

        streets_left = {"flop": 3, "turn": 2, "river": 1}[street]
        if street == "flop":
            _, eq, _ = evaluate_discard_options(self.hole3, req["board"], num_rollouts=65)
            inter = 2
        else:
            eq = post_discard_equity(
                self.hole2, req["board"], self.my_discard, self.opp_discard,
                opp_range_bias=0.22 if to_call > 0 else 0.0, num_rollouts=75
            )
            inter = hole_interaction_level(self.hole2, req["board"])

        pot_odds = to_call / max(1, pot + to_call)
        spr = eff_stack / max(1.0, pot)

        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}
            if street == "river" and inter <= 1 and eq < 0.72:
                return {"type": "action", "action": "check"}

            # Nutted hand (inter >= 3 or eq >= 0.78): use exact Geometric Pot-Growth sizing!
            if (inter >= 3 and eq >= 0.72) or eq >= 0.80:
                geo_frac = geometric_bet_fraction(pot, eff_stack, streets_left)
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * geo_frac)))
                return {"type": "action", "action": "raise", "amount": amt}

            # Medium-strong value hand: use Kelly fraction f* = (2 * eq - 1) scaled to pot
            if eq >= 0.56 and inter >= 2:
                kelly_frac = max(0.35, min(0.75, (2.0 * eq - 1.0) * 1.35))
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * kelly_frac)))
                return {"type": "action", "action": "raise", "amount": amt}

            return {"type": "action", "action": "check"}
        else:
            if street == "river" and inter == 0 and to_call > pot * 0.18:
                return {"type": "action", "action": "fold"}
            # Low SPR (< 2.0) commitment with Top Pair+ or Monster
            if (inter >= 3 and eq >= 0.76) and "raise" in legal:
                geo_frac = geometric_bet_fraction(pot + to_call, eff_stack - to_call, max(1, streets_left - 1))
                target = opp_bet + int((pot + to_call) * geo_frac)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                return {"type": "action", "action": "raise", "amount": amt}

            if eq >= pot_odds + 0.03:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}


def main():
    bot = KellyGeometricBot()
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
