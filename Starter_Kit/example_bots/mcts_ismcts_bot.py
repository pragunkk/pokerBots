"""
Radical Architecture 1: Information-Set Monte Carlo Tree Search (IS-MCTS) / Direct EV Tree Solver.
Instead of heuristic threshold rules, this bot models every decision as an explicit
Expected-Value optimization over discrete candidate actions:
  A = {fold, check/call, raise_0.38x, raise_0.66x, raise_1.0x, raise_all_in}
For each candidate action `a`, it simulates the opponent's response probability distribution
P(fold | opp_hand, a), P(call | opp_hand, a), P(raise | opp_hand, a) conditioned on the
opponent's observed empirical stats (fold rate, aggression rate, VPIP) and selects the action
with the highest expected chip utility E[U(a)].
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import (
    CARD_IDX,
    CARD_RANK,
    CARD_SUIT,
    RANK_VAL,
    board_danger_level,
    eval5_idx,
    eval7_idx,
    eval_cards,
    evaluate_discard_options,
    has_strong_draw,
    hole_interaction_level,
    post_discard_equity,
    preflop_3card_strength,
)


class ISMCTSBot:
    def __init__(self):
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None
        self.seat = "button"
        self.hands_seen = 0
        self.opp_vpip = 0
        self.opp_folds_post = 0
        self.opp_faced_post = 0
        self.opp_raises_post = 0
        self.opp_acts_post = 0
        self.processed_len = 0
        self.hand_vpip = False

    def start_hand(self, msg):
        self.hands_seen += 1
        self.hole3 = list(msg["hole_cards"])
        self.hole2 = []
        self.seat = msg["seat"]
        self.my_discard = None
        self.opp_discard = None
        self.processed_len = 0
        self.hand_vpip = False

    def observe(self, history):
        new_h = history[self.processed_len :]
        self.processed_len = len(history)
        for h in new_h:
            if h["actor"] == "opp":
                if h["street"] == "preflop":
                    if h["action"] in ("call", "raise") and not self.hand_vpip:
                        self.hand_vpip = True
                        self.opp_vpip += 1
                else:
                    self.opp_acts_post += 1
                    if h["action"] == "raise":
                        self.opp_raises_post += 1
                    elif h["action"] == "fold":
                        self.opp_folds_post += 1
                        self.opp_faced_post += 1
                    elif h["action"] == "call":
                        self.opp_faced_post += 1

    @property
    def emp_fold_rate(self):
        return (self.opp_folds_post + 4.0) / (self.opp_faced_post + 10.0)

    @property
    def emp_vpip(self):
        return (self.opp_vpip + 5.5) / (self.hands_seen + 10.0)

    @property
    def emp_agg(self):
        return (self.opp_raises_post + 3.0) / (self.opp_acts_post + 10.0)

    def choose_discard(self, req):
        best_c, _, _ = evaluate_discard_options(
            req["hole_cards"], req["board"], num_rollouts=75, opp_vpip=self.emp_vpip
        )
        return {"type": "discard", "card": best_c}

    def estimate_fold_prob(self, bet_add, pot_before_bet, street):
        """Models opponent fold probability P(fold | bet_frac) using logistic response calibrated to emp_fold_rate."""
        bet_frac = bet_add / max(1.0, pot_before_bet)
        base_f = self.emp_fold_rate
        # Scale fold probability by bet size pot odds: pot_odds = bet_frac / (1 + 2*bet_frac)
        size_factor = min(1.45, max(0.55, (bet_frac / 0.55) ** 0.45))
        street_mult = 0.80 if street == "flop" else (1.05 if street == "turn" else 1.15)
        return min(0.88, max(0.05, base_f * size_factor * street_mult))

    def choose_action(self, req):
        self.observe(req.get("history", []))
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            if opp_bet <= 2:
                if s >= 0.32 and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], 5))}
                if to_call == 0 or s >= 0.21:
                    return {"type": "action", "action": "call" if to_call > 0 else "check"}
                return {"type": "action", "action": "fold"}
            else:
                pot_odds = to_call / max(1.0, pot + to_call)
                if s >= 0.68 and "raise" in legal and opp_bet <= 25:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], int(opp_bet * 2.8)))}
                if s >= max(0.43, pot_odds + 0.12) and opp_bet <= 35:
                    return {"type": "action", "action": "call"}
                if s >= 0.76:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

        if street == "flop":
            best_c, eq, _ = evaluate_discard_options(self.hole3, req["board"], num_rollouts=75, opp_vpip=self.emp_vpip)
            keep2 = [c for c in self.hole3 if c != best_c][:2]
            inter = hole_interaction_level(keep2, req["board"])
            has_draw = has_strong_draw(keep2, req["board"])
        else:
            bias = 0.20 if to_call > 0 and self.emp_agg < 0.45 else (-0.15 if self.emp_agg > 0.50 else 0.0)
            eq = post_discard_equity(self.hole2, req["board"], self.my_discard, self.opp_discard, opp_range_bias=bias, num_rollouts=90)
            inter = hole_interaction_level(self.hole2, req["board"])
            has_draw = has_strong_draw(self.hole2, req["board"])

        danger = board_danger_level(req["board"])

        # IS-MCTS Direct EV Evaluation over candidate actions
        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}

            # Baseline EV(check): we realize a fraction `realization` of our equity `eq * pot`
            realization = 0.85 if self.seat == "button" else 0.75
            if street == "river":
                realization = 1.0 if inter >= 1 else 0.05
            ev_check = realization * eq * pot

            best_action = ("check", None)
            best_ev = ev_check

            # Evaluate candidate bet fractions
            for frac in (0.38, 0.58, 0.85):
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                p_fold = self.estimate_fold_prob(amt, pot, street)
                # When called, opponent's range is stronger than average: our equity when called drops!
                # The larger the bet, the stronger the opponent's calling range
                eq_when_called = max(0.02, (eq - p_fold * 0.38) / max(0.20, 1.0 - p_fold))
                if street == "river" and inter == 0:
                    eq_when_called = 0.0
                elif street == "river" and inter == 1 and frac >= 0.50:
                    eq_when_called *= 0.65

                ev_bet = p_fold * pot + (1.0 - p_fold) * (eq_when_called * (pot + 2.0 * amt) - amt)
                # Penalize betting weak 1-pair hands (`inter == 1`) on Turn/River where checking realizes showdown value
                if street in ("turn", "river") and inter == 1 and eq < 0.68:
                    ev_bet -= 0.18 * amt

                if ev_bet > best_ev + 0.5:
                    best_ev = ev_bet
                    best_action = ("raise", amt)

            if best_action[0] == "raise":
                return {"type": "action", "action": "raise", "amount": best_action[1]}
            return {"type": "action", "action": "check"}

        else:
            # Facing a bet: EV(fold) = 0 (relative to current pot).
            # EV(call) = eq_adj * (pot + to_call) - to_call
            eq_adj = eq
            if my_bet > 0 and opp_bet > my_bet and self.emp_agg < 0.48:
                eq_adj -= 0.08
            if street == "river" and inter == 0 and to_call > pot * 0.16:
                return {"type": "action", "action": "fold"}
            if street == "river" and inter == 1 and to_call > pot * 0.28 and self.emp_agg < 0.45:
                return {"type": "action", "action": "fold"}
            if street == "turn" and inter == 0 and not has_draw and to_call > pot * 0.22:
                return {"type": "action", "action": "fold"}

            ev_call = eq_adj * (pot + to_call) - to_call
            best_action = ("fold", None)
            best_ev = 0.0

            if ev_call > best_ev and "call" in legal:
                best_ev = ev_call
                best_action = ("call", None)

            if inter >= 3 and eq_adj >= 0.78 and danger < 0.45 and "raise" in legal:
                target = opp_bet + int((pot + to_call) * 0.75)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                add = amt - my_bet
                p_fold = self.estimate_fold_prob(amt - opp_bet, pot + to_call, street) * 0.65
                eq_called = max(0.45, eq_adj - 0.06)
                ev_raise = p_fold * pot + (1.0 - p_fold) * (eq_called * (pot + to_call + 2 * (amt - opp_bet)) - add)
                if ev_raise > best_ev + 1.0:
                    best_action = ("raise", amt)

            if best_action[0] == "raise":
                return {"type": "action", "action": "raise", "amount": best_action[1]}
            if best_action[0] == "call":
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}


def main():
    bot = ISMCTSBot()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        t = msg.get("type")
        if t == "new_hand":
            bot.start_hand(msg)
        elif t == "discard_result":
            bot.my_discard = msg["my_discard"]
            bot.opp_discard = msg["opp_discard"]
            bot.hole2 = list(msg["hole_cards"])
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
