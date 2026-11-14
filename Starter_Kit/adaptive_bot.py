"""
Adaptive Meta-Strategy Bot v8 (Championship Submission for Official Two-Leg Discard Hold'em).
Key Upgrades in v8 (building on v7 Soft Preflop RangeWeight & v6.2 MDF/Board-Interaction Engine):
1. Startup O(1) Lookup-Table Hand Evaluator (POW5 + Flush Bitmask in `fast_eval.py`):
   - Precomputes `NON_FLUSH_LUT`, `FLUSH_BIT_LUT`, `N6`, `N7`, `FL`, and `FSUIT` during the 5s startup
     window, accelerating 5/6/7-card hand evaluation by ~10x (< 1.5ms max decision latency, ~0.25s per
     400-hand match) and enabling higher Flop (`280 -> 45`) and Turn (`380 -> 60`) rollouts.
2. Soft Sigmoid Preflop Range Weighting (`RangeWeight`):
   - Weights opponent 3-card holdings `(opp0, opp1, opp_disc)` by smooth logistic bounds `(lo, hi)` derived
     from their empirical preflop frequencies (`btn_open_rate`, `pfr`, `three_bet_vs_open_rate`, `fold_to_steal`).
3. True 8-Out Straight Draw Filter (`has_strong_draw`) & Flop Out-of-Position Donk-Bet Guard:
   - Distinguishes 8-out open-ended/double-gutshot draws from 4-out single gutshots (`L1 #148`, `L2 #54`, `L2 #8`),
     preventing loose flop floats with bare 4-out gutshots that auto-fold on the turn.
   - Stops out-of-position flop donk-betting (`self.seat == "big_blind"` and `opp_was_prev_aggressor`) with
     bluffs and weak/medium 1-pair hands (`L1 #60`, `L1 #156`, `L2 #5`, `L2 #9`).
4. Street-Specific Flop Over-Folder Exploitation, MDF-Balanced Turn Calibration & River Overbet Guard:
   - Tracks `fold_to_flop_bet` and `turn_barrel_rate` independently. Against fit-or-fold flop players
     (`fold_to_flop_bet > 0.44`, like `spar_bot.py` which folds 600+ flops), probes `0.35x` pot at high
     frequency (`68%`) when checked to on the flop, while maintaining Minimum Defense Frequency (`< 42%`
     postflop fold rate) on the Flop and Turn (`turn_inter1_eq = 0.53` vs `turn_barrel_rate > 0.36`).
   - Prevents false `MANIAC` classification from preflop 3-betting alone without postflop mania, and folds
     bare One-Pair/Overpair (`inter <= 2`) to massive river overbet jams (`to_call >= 60` and `>= 0.85x` pot).
"""

import json
import random
import sys
import traceback
from collections import deque
from fast_eval import (
    RANK_VAL,
    RangeWeight,
    board_danger_level,
    eval_cards,
    evaluate_discard_options,
    has_strong_draw,
    hole_interaction_level,
    init_fast_tables,
    top_range_equity,
    post_discard_equity,
    preflop_3card_strength,
    strength_quantile,
)


def geometric_bet_fraction(pot, eff_stack, streets_left):
    if pot <= 0 or eff_stack <= 0 or streets_left <= 0:
        return 0.66
    target_ratio = (2.0 * eff_stack + pot) / pot
    frac = 0.5 * ((target_ratio ** (1.0 / streets_left)) - 1.0)
    return max(0.45, min(1.15, frac))


class OpponentProfile:
    def __init__(self, window_size=50):
        self.hands_seen = 0
        self.vpip_hands = 0
        self.pfr_hands = 0
        self.three_bets = 0
        self.preflop_shoves = 0
        self.faced_preflop_steal = 0
        self.folded_to_preflop_steal = 0
        self.three_bet_my_open = 0
        self.opp_btn_hands = 0
        self.opp_btn_opens = 0

        self.postflop_bets_raises = 0
        self.postflop_calls_checks = 0
        self.faced_bet_postflop = 0
        self.folded_to_bet_postflop = 0
        self.faced_flop_bet = 0
        self.folded_to_flop_bet = 0
        self.faced_turn_river_bet = 0
        self.folded_to_turn_river_bet = 0
        self.raised_my_bet_postflop = 0
        self.check_raises = 0
        self.turn_bet_hands = 0
        self.turn_seen_hands = 0
        self.overbets = 0            # very large turn/river bets (>= 1.1x pot) we have faced
        self.overbet_showdowns = 0   # ...that went to showdown
        self.overbet_bluffs = 0      # ...where they showed a weak hand
        self.hand_overbet_streets = set()
        self.river_bet_hands = 0
        self.river_seen_hands = 0

        self.high_card_discards = 0
        self.total_discards = 0
        self.showdowns = 0
        self.showdown_bluffs = 0

        self.recent_window = deque(maxlen=window_size)
        self.reset_hand_flags()

    def reset_hand_flags(self):
        self.hand_vpip = 0
        self.hand_pfr = 0
        self.hand_shoved_pre = 0
        self.hand_agg = 0
        self.hand_pass = 0
        self.hand_folded = 0
        self.hand_faced = 0
        self.hand_raised_me = 0
        self.hand_saw_turn = False
        self.hand_bet_turn = False
        self.hand_overbet_streets = set()
        self.hand_saw_river = False
        self.hand_bet_river = False
        self.opp_checked_this_street = False
        self.me_bet_this_street = False
        self.me_opened_preflop = False
        self.opp_responded_to_open = False
        self.opp_first_pre_tracked = False
        self.last_street_tracked = "preflop"
        self.processed_history_len = 0

    def start_hand(self, my_seat="button"):
        if self.hands_seen > 0:
            if self.hand_saw_turn:
                self.turn_seen_hands += 1
                if self.hand_bet_turn:
                    self.turn_bet_hands += 1
            if self.hand_saw_river:
                self.river_seen_hands += 1
                if self.hand_bet_river:
                    self.river_bet_hands += 1
            self.recent_window.append(
                (
                    self.hand_vpip,
                    self.hand_pfr,
                    self.hand_agg,
                    self.hand_pass,
                    self.hand_folded,
                    self.hand_faced,
                    self.hand_raised_me,
                )
            )
        self.hands_seen += 1
        self.reset_hand_flags()
        self.opp_is_btn = (my_seat != "button")
        if self.opp_is_btn:
            self.opp_btn_hands += 1

    def observe_history(self, history):
        new_entries = history[self.processed_history_len :]
        self.processed_history_len = len(history)

        preflop_raises_seen = 0
        for entry in history[: self.processed_history_len - len(new_entries)]:
            if entry["street"] == "preflop" and entry["action"] == "raise":
                preflop_raises_seen += 1

        for entry in new_entries:
            street = entry["street"]
            actor = entry["actor"]
            action = entry["action"]
            amount = entry.get("amount", 0) or 0

            if street == "turn":
                self.hand_saw_turn = True
            elif street == "river":
                self.hand_saw_river = True

            if street != self.last_street_tracked:
                self.opp_checked_this_street = False
                self.me_bet_this_street = False
                self.last_street_tracked = street

            if actor == "opp":
                if street == "preflop":
                    if self.opp_is_btn and not self.opp_first_pre_tracked:
                        self.opp_first_pre_tracked = True
                        if action == "raise":
                            self.opp_btn_opens += 1
                    if self.me_opened_preflop and not self.opp_responded_to_open:
                        self.opp_responded_to_open = True
                        self.faced_preflop_steal += 1
                        if action == "fold":
                            self.folded_to_preflop_steal += 1
                        elif action == "raise":
                            self.three_bet_my_open += 1
                    if action in ("call", "raise") and not self.hand_vpip:
                        self.hand_vpip = 1
                        self.vpip_hands += 1
                    if action == "raise":
                        if not self.hand_pfr:
                            self.hand_pfr = 1
                            self.pfr_hands += 1
                        if amount >= 50 and not self.hand_shoved_pre:
                            self.hand_shoved_pre = 1
                            self.preflop_shoves += 1
                        if preflop_raises_seen >= 1:
                            self.three_bets += 1
                        preflop_raises_seen += 1
                else:
                    if action == "check":
                        self.opp_checked_this_street = True
                        self.postflop_calls_checks += 1
                        self.hand_pass += 1
                    elif action == "call":
                        self.postflop_calls_checks += 1
                        self.hand_pass += 1
                    elif action == "raise":
                        self.postflop_bets_raises += 1
                        self.hand_agg += 1
                        if street == "turn":
                            self.hand_bet_turn = True
                        elif street == "river":
                            self.hand_bet_river = True
                        if self.me_bet_this_street:
                            self.raised_my_bet_postflop += 1
                            self.hand_raised_me += 1
                        if self.opp_checked_this_street:
                            self.check_raises += 1
                    elif action == "fold":
                        self.folded_to_bet_postflop += 1
                        self.hand_folded += 1
                        if street == "flop":
                            self.folded_to_flop_bet += 1
                        else:
                            self.folded_to_turn_river_bet += 1

            elif actor == "me":
                if street == "preflop" and action == "raise":
                    if preflop_raises_seen == 0:
                        self.me_opened_preflop = True
                    preflop_raises_seen += 1
                elif street != "preflop" and action == "raise":
                    self.faced_bet_postflop += 1
                    self.hand_faced += 1
                    self.me_bet_this_street = True
                    if street == "flop":
                        self.faced_flop_bet += 1
                    else:
                        self.faced_turn_river_bet += 1

    def observe_discard(self, opp_discard):
        if opp_discard:
            self.total_discards += 1
            if RANK_VAL[opp_discard[0]] >= 11:
                self.high_card_discards += 1

    @property
    def vpip(self):
        g = (self.vpip_hands + 5.5) / (self.hands_seen + 10.0)
        if len(self.recent_window) >= 18:
            r = sum(x[0] for x in self.recent_window) / len(self.recent_window)
            return 0.45 * g + 0.55 * r
        return g

    @property
    def pfr(self):
        g = (self.pfr_hands + 3.0) / (self.hands_seen + 10.0)
        if len(self.recent_window) >= 18:
            r = sum(x[1] for x in self.recent_window) / len(self.recent_window)
            return 0.45 * g + 0.55 * r
        return g

    @property
    def btn_open_rate(self):
        return (self.opp_btn_opens + 3.0) / (self.opp_btn_hands + 6.0)

    @property
    def fold_to_steal(self):
        return (self.folded_to_preflop_steal + 2.0) / (self.faced_preflop_steal + 5.0)

    @property
    def three_bet_vs_open_rate(self):
        return (self.three_bet_my_open + 0.5) / (self.faced_preflop_steal + 5.0)

    @property
    def preflop_shove_rate(self):
        return self.preflop_shoves / max(1.0, float(self.hands_seen))

    @property
    def fold_to_bet(self):
        g = (self.folded_to_bet_postflop + 4.0) / (self.faced_bet_postflop + 10.0)
        if len(self.recent_window) >= 18:
            r_faced = sum(x[5] for x in self.recent_window)
            if r_faced >= 6:
                r = sum(x[4] for x in self.recent_window) / r_faced
                return 0.40 * g + 0.60 * r
        return g

    @property
    def fold_to_flop_bet(self):
        return (self.folded_to_flop_bet + 2.5) / (self.faced_flop_bet + 6.0)

    @property
    def fold_to_turn_river_bet(self):
        return (self.folded_to_turn_river_bet + 2.0) / (self.faced_turn_river_bet + 6.0)

    @property
    def raise_my_bet_rate(self):
        return (self.raised_my_bet_postflop + 1.0) / (self.faced_bet_postflop + 6.0)

    @property
    def postflop_agg_per_hand(self):
        g = (self.postflop_bets_raises + 3.0) / (self.hands_seen + 10.0)
        if len(self.recent_window) >= 18:
            r = sum(x[2] for x in self.recent_window) / len(self.recent_window)
            return 0.40 * g + 0.60 * r
        return g

    @property
    def overbet_rate(self):
        """How often they make a very large bet, per turn/river street they reach."""
        return (self.overbets + 0.4) / (self.turn_seen_hands + self.river_seen_hands + 8.0)

    @property
    def overbet_bluff_rate(self):
        """Share of their very large bets that were weak hands at showdown (prior: 10%)."""
        return (self.overbet_bluffs + 0.5) / (self.overbet_showdowns + 5.0)

    @property
    def turn_barrel_rate(self):
        return (self.turn_bet_hands + 1.5) / (self.turn_seen_hands + 5.0)

    @property
    def river_barrel_rate(self):
        return (self.river_bet_hands + 1.5) / (self.river_seen_hands + 5.0)

    @property
    def showdown_bluff_rate(self):
        return (self.showdown_bluffs + 0.4) / (self.showdowns + 3.5)

    def classify(self):
        if self.hands_seen < 8:
            return "BALANCED"

        three_bet_rate = self.three_bets / max(1, self.hands_seen)
        if (
            self.preflop_shove_rate > 0.25
            or self.postflop_agg_per_hand > 1.15
            or (three_bet_rate > 0.22 and self.postflop_agg_per_hand > 0.85)
            or (self.pfr > 0.62 and self.postflop_agg_per_hand > 0.90)
            or (self.showdowns >= 4 and self.showdown_bluff_rate > 0.42 and self.postflop_agg_per_hand > 0.75)
        ):
            return "MANIAC"

        if self.check_raises >= 3 and self.check_raises / max(1, self.hands_seen) > 0.055:
            return "TRAPPER"

        if self.fold_to_bet < 0.28 and self.vpip > 0.58 and self.postflop_agg_per_hand < 0.45:
            return "CALLING_STATION"

        if (
            self.vpip < 0.45
            or (self.fold_to_steal > 0.62 and self.three_bet_vs_open_rate < 0.12)
            or (
                self.fold_to_bet > 0.48
                and self.three_bet_vs_open_rate < 0.13
                and self.turn_barrel_rate < 0.35
            )
        ):
            return "NIT_TAG"

        return "BALANCED"


class AdaptiveBot:
    def __init__(self):
        self.total_hands = 200  # Per-leg hands (200 in Stage 1, 500 in Stage 2)
        self.starting_stack = 200
        self.time_bank_ms = 30000  # Per-leg time bank (30s in Stage 1, 75s in Stage 2)
        self.cap = 400

        self.hand_id = 0
        self.seat = "button"
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None
        self.cum_net = 0

        self.opp = OpponentProfile()
        self.cached_flop_discard = None
        self.cached_flop_eq = None
        self.cached_flop_eq_map = None
        self.pre_w = None

    def handle_init(self, msg):
        self.total_hands = msg.get("total_hands", 200)
        self.starting_stack = msg.get("starting_stack", 200)
        self.time_bank_ms = msg.get("time_bank_ms", 30000)
        self.cap = 1000 if self.total_hands >= 500 else 400
        init_fast_tables()
        strength_quantile(0.5)  # build the strength distribution now, while the clock is not running

    def handle_new_hand(self, msg):
        self.hand_id = msg["hand_id"]
        self.seat = msg["seat"]
        self.hole3 = list(msg["hole_cards"])
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None
        self.cached_flop_discard = None
        self.cached_flop_eq = None
        self.cached_flop_eq_map = None
        self.pre_w = None
        self.opp.start_hand(self.seat)

    def handle_discard_result(self, msg):
        self.my_discard = msg["my_discard"]
        self.opp_discard = msg["opp_discard"]
        self.hole2 = list(msg["hole_cards"])
        self.opp.observe_discard(self.opp_discard)

    def handle_hand_result(self, msg):
        self.cum_net += msg.get("net", 0)
        hist = msg.get("history", [])
        self.opp.observe_history(hist)
        if msg.get("result") == "showdown" and msg.get("opp_hole_cards"):
            opp_bet_late = any(
                h["actor"] == "opp" and h["street"] in ("turn", "river") and h["action"] == "raise"
                for h in hist
            )
            if self.opp.hand_overbet_streets:
                self.opp.overbet_showdowns += 1
                if hole_interaction_level(msg["opp_hole_cards"], msg["board"]) <= 1:
                    self.opp.overbet_bluffs += 1
            if opp_bet_late:
                self.opp.showdowns += 1
                inter = hole_interaction_level(msg["opp_hole_cards"], msg["board"])
                if inter <= 1:
                    self.opp.showdown_bluffs += 1

    def opp_preflop_range(self, hist):
        """Soft model of the opponent's starting-hand range from how they played preflop this hand.
        Thresholds come from their own measured frequencies, mapped onto the strength distribution
        of all 22,100 starting hands (top x% of hands -> strength_quantile(1 - x))."""
        pre = [h for h in hist if h.get("street") == "preflop"]
        opp_acts = [h["action"] for h in pre if h.get("actor") == "opp"]
        if not opp_acts:
            return None
        my_raises = sum(1 for h in pre if h.get("actor") == "me" and h.get("action") == "raise")
        opp_raises = opp_acts.count("raise")
        o = self.opp

        def top(x):  # strength above which the top fraction x of hands lies
            return strength_quantile(1.0 - max(0.04, min(0.96, x)))

        if self.seat == "big_blind":  # opponent is the button and acted first
            if opp_acts[0] == "raise":
                if opp_raises >= 2:                      # opened, then 4-bet our 3-bet
                    return RangeWeight(lo=top(0.07))
                if my_raises >= 1:                       # opened, then called our 3-bet
                    return RangeWeight(lo=top(o.btn_open_rate * 0.55))
                return RangeWeight(lo=top(o.btn_open_rate))
            if opp_acts[0] == "call":                    # limped
                if opp_raises >= 1:                      # limp then raise
                    return RangeWeight(lo=top(0.25))
                if o.opp_btn_hands >= 10 and o.btn_open_rate > 0.15:
                    return RangeWeight(hi=top(o.btn_open_rate))  # would usually have opened strong hands
            return None
        # we are the button
        if my_raises == 0:                               # we limped
            if opp_raises >= 1:
                return RangeWeight(lo=top(max(0.15, o.pfr)))
            return RangeWeight(hi=top(max(0.10, o.pfr * 0.8)))  # checked: strong hands would have raised
        if opp_acts[0] == "raise":                       # 3-bet our open
            if opp_raises >= 2:
                return RangeWeight(lo=top(0.05))
            if my_raises >= 2:                           # 3-bet, we 4-bet, they called
                return RangeWeight(lo=top(o.three_bet_vs_open_rate * 0.6))
            return RangeWeight(lo=top(o.three_bet_vs_open_rate))
        if opp_acts[0] == "call":                        # defended the big blind
            return RangeWeight(lo=top(1.0 - o.fold_to_steal), hi=top(o.three_bet_vs_open_rate))
        return None

    def rollouts(self, full, minimum, time_left):
        """Time-scaled rollouts calibrated for 2-leg per-leg time banks (powered by O(1) lookup tables)."""
        hands_left = max(1, self.total_hands - self.hand_id + 1)
        per_hand_ms = time_left / hands_left
        scale = max(0.0, min(1.0, (per_hand_ms - 18.0) / 35.0))
        return int(minimum + (full - minimum) * scale)

    def choose_discard(self, req):
        hole = req["hole_cards"]
        board = req["board"]
        if self.cached_flop_discard and set(hole) == set(self.hole3):
            best_card = self.cached_flop_discard
        else:
            time_left = req.get("time_left_ms", 30000)
            rollouts = self.rollouts(520, 80, time_left)
            best_card, _, _ = evaluate_discard_options(
                hole, board, num_rollouts=rollouts, opp_vpip=self.opp.vpip, pre_w=self.pre_w
            )
        return {"type": "discard", "card": best_card}

    def choose_action(self, req):
        self.opp.observe_history(req.get("history", []))
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        time_left = req.get("time_left_ms", 30000)

        archetype = self.opp.classify()

        if time_left < 2000:
            if "check" in legal:
                return {"type": "action", "action": "check"}
            if to_call <= 4 and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if street == "preflop":
            return self._play_preflop(req, archetype)
        else:
            return self._play_postflop(req, archetype, time_left)

    def _play_preflop(self, req, archetype):
        legal = req["legal_actions"]
        to_call = req["to_call"]
        opp_bet = req["opp_bet"]
        my_bet = req["my_bet"]
        strength = preflop_3card_strength(self.hole3)

        # Special Case 1: Opponent is a Preflop All-In Shove Spammer (`preflop_shove_rate > 0.25`)
        if self.opp.hands_seen >= 6 and self.opp.preflop_shove_rate > 0.25:
            shove_call_t = 0.57
            if strength >= shove_call_t:
                if "raise" in legal and strength >= 0.72:
                    return {"type": "action", "action": "raise", "amount": req["max_raise"]}
                if "call" in legal and to_call > 0:
                    return {"type": "action", "action": "call"}
                if "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": min(req["max_raise"], 6)}
            if to_call == 0:
                return {"type": "action", "action": "check"}
            return {"type": "action", "action": "fold"}

        # Special Case 2: Opponent is an Extreme Preflop Folder (`fold_to_steal > 0.62`)
        if self.opp.hands_seen >= 8 and self.opp.fold_to_steal > 0.62 and self.opp.three_bet_vs_open_rate < 0.12 and opp_bet <= 2:
            if "raise" in legal:
                return {"type": "action", "action": "raise", "amount": req["min_raise"]}

        open_thresh = 0.32
        limp_thresh = 0.21
        call_raise_thresh = 0.37
        three_bet_val_thresh = 0.66

        if archetype == "NIT_TAG":
            open_thresh = 0.23
            call_raise_thresh = 0.45
            three_bet_val_thresh = 0.71
        elif archetype == "CALLING_STATION":
            open_thresh = 0.38
            limp_thresh = 0.22
        elif archetype == "MANIAC":
            open_thresh = 0.36
            call_raise_thresh = 0.35
            three_bet_val_thresh = 0.65

        # Anti-3-Bet Button Limp/Raise Mix:
        # When an opponent 3-bets our Button opens frequently (> 15%), stop open-raising weak 0.32-0.44 hands
        # into their 3-bets (which loses 5-6 chips on a fold). Instead, limp (call 1 chip) with 0.20-0.44
        # and only open-raise hands strong enough to defend (`>= 0.44`)!
        if self.opp.faced_preflop_steal >= 6 and self.opp.three_bet_vs_open_rate > 0.115:
            open_thresh = max(open_thresh, 0.44)
            limp_thresh = 0.20

        # Active 3-Bet & BB Defense vs Wide Button Openers:
        if self.seat != "button" and self.opp.opp_btn_hands >= 8 and self.opp.btn_open_rate > 0.56:
            three_bet_val_thresh = min(three_bet_val_thresh, 0.58)
            call_raise_thresh = min(call_raise_thresh, 0.34)

        if opp_bet <= 2:
            if strength >= open_thresh and "raise" in legal:
                if archetype == "CALLING_STATION" and strength > 0.60:
                    size = 7
                elif self.opp.faced_preflop_steal >= 6 and self.opp.three_bet_vs_open_rate > 0.115:
                    size = 5
                elif strength >= 0.56:
                    size = 6
                else:
                    size = 5
                amt = max(req["min_raise"], min(req["max_raise"], size))
                return {"type": "action", "action": "raise", "amount": amt}
            if to_call == 0:
                return {"type": "action", "action": "check"}
            if strength >= limp_thresh and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        pot = req["pot"]
        pot_odds = to_call / max(1, pot + to_call)

        if opp_bet >= 35:
            req_s = 0.76 if archetype != "MANIAC" else 0.62
            if strength >= req_s:
                if strength >= 0.85 and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": req["max_raise"]}
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if archetype == "MANIAC" and my_bet >= 5:
            if strength >= 0.74 and "raise" in legal:
                target = min(req["max_raise"], max(req["min_raise"], opp_bet * 3))
                return {"type": "action", "action": "raise", "amount": target}
            if strength >= 0.40:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if my_bet >= 5 and opp_bet >= 14:
            if strength >= 0.80 and "raise" in legal:
                target = min(req["max_raise"], max(req["min_raise"], opp_bet * 3))
                return {"type": "action", "action": "raise", "amount": target}
            if strength >= 0.49 or (self.seat == "button" and strength >= 0.46 and pot_odds <= 0.34):
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if strength >= three_bet_val_thresh and "raise" in legal:
            target = min(req["max_raise"], max(req["min_raise"], int(opp_bet * 2.8)))
            return {"type": "action", "action": "raise", "amount": target}

        btn_ip_discount = 0.04 if (self.seat == "button" and self.opp.three_bet_vs_open_rate > 0.11) else 0.0
        if strength >= (call_raise_thresh - btn_ip_discount) or (pot_odds <= 0.34 and strength >= 0.33):
            return {"type": "action", "action": "call"}

        return {"type": "action", "action": "fold"}

    def _play_postflop(self, req, archetype, time_left):
        legal = req["legal_actions"]
        street = req["street"]
        board = req["board"]
        to_call = req["to_call"]
        pot = req["pot"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        eff_stack = min(req["my_stack"], req["opp_stack"])
        hist = req.get("history", [])

        if self.pre_w is None:
            self.pre_w = self.opp_preflop_range(hist)
        pre_raises = [h for h in hist if h["street"] == "preflop" and h["action"] == "raise"]
        is_3bet_pot = len(pre_raises) >= 2
        freq_raiser = self.opp.faced_bet_postflop >= 5 and self.opp.raise_my_bet_rate > 0.20

        # v8 Profile Separation:
        # Detect opponents who barrel the Turn frequently (`turn_barrel_rate > 0.36`, like `spar_bot.py`)
        # or play high-aggression postflop (`postflop_agg_per_hand > 0.58`), so we maintain Minimum Defense
        # Frequency (MDF, keeping our postflop fold rate < 42%) and prevent opponent over-folder exploits!
        agg_turn_opp = self.opp.turn_seen_hands >= 8 and self.opp.turn_barrel_rate > 0.36
        agg_postflop_opp = (
            archetype == "MANIAC"
            or (self.opp.hands_seen >= 10 and self.opp.postflop_agg_per_hand > 0.58)
            or freq_raiser
            or (street == "turn" and agg_turn_opp)
        )

        prev_street = {"flop": "preflop", "turn": "flop", "river": "turn"}[street]
        prev_acts = [h for h in hist if h["street"] == prev_street]
        prev_raises = [h for h in prev_acts if h["action"] == "raise"]
        opp_was_prev_aggressor = bool(prev_raises and prev_raises[-1]["actor"] == "opp")
        prev_checked_through = (street in ("turn", "river")) and len(prev_raises) == 0

        opp_bet_flop = any(h["street"] == "flop" and h["actor"] == "opp" and h["action"] == "raise" for h in hist)
        opp_bet_turn = any(h["street"] == "turn" and h["actor"] == "opp" and h["action"] == "raise" for h in hist)
        opp_raised_us_flop = False
        flop_r = [h["actor"] for h in hist if h["street"] == "flop" and h["action"] == "raise"]
        if len(flop_r) >= 2 and "me" in flop_r and flop_r[-1] == "opp":
            opp_raised_us_flop = True

        if street == "flop":
            if self.cached_flop_eq is None:
                rollouts = self.rollouts(520, 80, time_left)
                best_c, best_eq, eq_map = evaluate_discard_options(
                    self.hole3, board, num_rollouts=rollouts, opp_vpip=self.opp.vpip, pre_w=self.pre_w
                )
                self.cached_flop_discard = best_c
                self.cached_flop_eq = best_eq
                self.cached_flop_eq_map = eq_map
            eq = self.cached_flop_eq
            keep2 = [c for c in self.hole3 if c != self.cached_flop_discard][:2]
            inter = hole_interaction_level(keep2, board)
            full_cat = eval_cards(keep2, board) >> 20
            has_draw = has_strong_draw(keep2, board)
        else:
            rollouts = self.rollouts(720, 110, time_left)
            if archetype == "MANIAC":
                opp_bias = -0.20 if to_call > 0 else 0.0
            elif agg_postflop_opp and to_call > 0 and not is_3bet_pot:
                opp_bias = 0.06
            elif archetype == "NIT_TAG" and to_call > 0:
                opp_bias = 0.24
            else:
                opp_bias = 0.16 if (to_call > 0 or opp_raised_us_flop) else 0.0

            eq = post_discard_equity(
                self.hole2,
                board,
                self.my_discard,
                self.opp_discard,
                opp_range_bias=opp_bias,
                num_rollouts=rollouts,
                pre_w=self.pre_w,
            )
            inter = hole_interaction_level(self.hole2, board)
            full_cat = eval_cards(self.hole2, board) >> 20
            has_draw = has_strong_draw(self.hole2, board)

            if self.opp_discard and to_call > 0 and not agg_postflop_opp:
                disc_r = RANK_VAL[self.opp_discard[0]]
                if disc_r >= 12:
                    eq *= 0.95

        danger = board_danger_level(board)
        pot_odds = to_call / max(1, pot + to_call)
        eq_adj = eq

        # ------------------------------------------------------------------
        # Scenario A: Checked to us (to_call == 0)
        # ------------------------------------------------------------------
        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}

            if self.opp.hands_seen >= 10 and self.opp.fold_to_bet > 0.68:
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.30)))
                return {"type": "action", "action": "raise", "amount": amt}

            # Rule 1a (v8 Fix): Out of position on the Flop (`self.seat == "big_blind"`), do not donk-bet
            # weak/medium hands or bluffs (`inter < 3` and `eq_adj < 0.70`) into the Button preflop raiser!
            if street == "flop" and self.seat != "button" and opp_was_prev_aggressor and inter < 3 and eq_adj < 0.70:
                return {"type": "action", "action": "check"}

            # Rule 1b: Do not "donk-bet" into the previous street's raiser on Turn/River without Two Pair+ (`inter >= 3`)
            if street in ("turn", "river") and opp_was_prev_aggressor:
                if inter < 3 and (eq_adj < 0.78 or opp_raised_us_flop):
                    return {"type": "action", "action": "check"}
                if street == "river" and opp_raised_us_flop and inter == 3 and full_cat == 2 and eq_adj < 0.84:
                    return {"type": "action", "action": "check"}

            # Rule 2a (v9): OOP Turn Check-Raise Trap with monsters vs frequent turn barrelers
            if (
                street == "turn"
                and self.seat != "button"
                and inter >= 3
                and eq_adj >= 0.78
                and self.opp.turn_barrel_rate > 0.38
                and random.random() < 0.35
            ):
                return {"type": "action", "action": "check"}

            # Rule 2: Against a MANIAC, check-call medium hands on Flop/Turn so they bluff into us; bet big with monsters
            if archetype == "MANIAC":
                if street in ("flop", "turn") and 0.48 <= eq_adj <= 0.75:
                    return {"type": "action", "action": "check"}
                if eq_adj > 0.75 or (street == "river" and eq_adj >= 0.58 and inter >= 2):
                    frac = 0.85 if eq_adj > 0.80 else 0.62
                    target = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                    return {"type": "action", "action": "raise", "amount": target}
                return {"type": "action", "action": "check"}

            # Rule 3: Pot-control weak/underpairs (`inter <= 1`) on Flop (vs frequent check-raisers / 3-bet pots), Turn & River!
            if street == "flop" and (freq_raiser or is_3bet_pot) and inter <= 1 and not has_draw and eq_adj < 0.65:
                return {"type": "action", "action": "check"}

            if street == "river" and inter <= 1 and eq_adj < 0.74:
                if (
                    inter == 0
                    and eq_adj < 0.28
                    and prev_checked_through
                    and archetype not in ("CALLING_STATION", "MANIAC")
                ):
                    if random.random() < 0.45:
                        target = max(req["min_raise"], min(req["max_raise"], int(pot * 0.38) + 1))
                        return {"type": "action", "action": "raise", "amount": target}
                return {"type": "action", "action": "check"}

            if street == "turn" and inter == 1 and eq_adj < 0.66:
                return {"type": "action", "action": "check"}

            # Rule 4: Asymmetric Value Sizing:
            if inter >= 3 and eq_adj >= 0.72:
                streets_left = {"flop": 3, "turn": 2, "river": 1}[street]
                geo_frac = geometric_bet_fraction(pot, eff_stack, streets_left)
                frac = max(0.68, min(0.95, geo_frac))
                if archetype == "CALLING_STATION":
                    frac = max(frac, 0.88)
                target = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                return {"type": "action", "action": "raise", "amount": target}

            val_thresh = 0.56
            if archetype == "CALLING_STATION":
                val_thresh = 0.56 if street == "flop" else 0.60
            elif archetype == "TRAPPER" or freq_raiser:
                val_thresh = 0.61

            if eq_adj >= val_thresh:
                if street == "river" and inter == 2 and danger >= 0.55 and eq_adj < 0.66 and archetype != "CALLING_STATION":
                    return {"type": "action", "action": "check"}
                if archetype == "CALLING_STATION":
                    frac = 0.68
                elif inter >= 2 and eq_adj >= 0.68:
                    frac = 0.62
                else:
                    frac = 0.52
                target = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                return {"type": "action", "action": "raise", "amount": target}

            # Rule 5: Small-Ball Probe & Delayed Turn Float (`META_HUNTER` Exploit)
            bluff_freq = 0.0
            probe_frac = 0.36
            if not freq_raiser and archetype not in ("CALLING_STATION", "MANIAC", "TRAPPER"):
                if street == "flop" and 0.32 <= eq_adj < val_thresh and not is_3bet_pot:
                    if self.opp.faced_flop_bet >= 5 and self.opp.fold_to_flop_bet > 0.44:
                        bluff_freq = 0.68
                        probe_frac = 0.35
                    elif eq_adj >= 0.38:
                        bluff_freq = 0.52 if archetype == "NIT_TAG" else 0.36
                        probe_frac = 0.36
                elif street == "turn" and prev_checked_through and 0.30 <= eq_adj < val_thresh:
                    bluff_freq = 0.48 if archetype == "NIT_TAG" else 0.38
                    probe_frac = 0.40

            if bluff_freq > 0 and random.random() < bluff_freq:
                target = max(req["min_raise"], min(req["max_raise"], int(pot * probe_frac)))
                return {"type": "action", "action": "raise", "amount": target}

            return {"type": "action", "action": "check"}

        # ------------------------------------------------------------------
        # Scenario B: Facing a Bet or Raise (to_call > 0)
        # ------------------------------------------------------------------
        street_acts = [h for h in hist if h["street"] == street]
        opp_check_raised = (
            len(street_acts) >= 3
            and street_acts[0]["actor"] == "opp"
            and street_acts[0]["action"] == "check"
            and street_acts[-1]["actor"] == "opp"
            and street_acts[-1]["action"] == "raise"
        )
        opp_raised_our_bet = my_bet > 0 and opp_bet > my_bet
        bet_to_pot_ratio = to_call / max(1.0, pot - to_call)

        # 0. Very large bets (>= 1.1x pot) on the turn or river come from the top of a range.
        # Compare our hand with only the strongest slice of their possible hands (slice size = how
        # often this opponent makes such bets) plus their measured bluff share, instead of with
        # their whole range. Learned per opponent, so frequent or bluffy overbettors get called more.
        if street in ("turn", "river") and bet_to_pot_ratio >= 1.1 and to_call >= 20 and self.opp_discard:
            if street not in self.opp.hand_overbet_streets:
                self.opp.hand_overbet_streets.add(street)
                self.opp.overbets += 1
            q = max(0.05, min(0.60, self.opp.overbet_rate))
            eq_top = top_range_equity(self.hole2, board, self.my_discard, self.opp_discard, q, self.pre_w)
            if street == "turn" and has_draw:
                eq_top = max(eq_top, 0.18)
            beta = min(0.60, self.opp.overbet_bluff_rate)
            p_win = beta + (1.0 - beta) * eq_top
            if eq_top < 0.85:
                if p_win >= pot_odds + 0.03 and "call" in legal:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold" if "fold" in legal else "check"}

        # 0b. Price floor for small bets (<= 34% pot, not a raise over our bet). The fold rules below
        # look at hand type and opponent history but not at the price, so a 2-chip bet into 16 used
        # to fold out pairs and draws that need only ~11% equity. Against a tiny bet: raise strong
        # hands for value, otherwise decide purely on equity versus pot odds.
        if bet_to_pot_ratio <= 0.34 and not opp_raised_our_bet and not opp_check_raised:
            if inter >= 3 and eq_adj >= 0.80 and "raise" in legal:
                target = opp_bet + int((pot + to_call) * 0.70)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                return {"type": "action", "action": "raise", "amount": amt}
            need = pot_odds + (0.03 if bet_to_pot_ratio <= 0.25 else 0.08)
            if eq >= need and "call" in legal:
                return {"type": "action", "action": "call"}
            if bet_to_pot_ratio <= 0.25:
                return {"type": "action", "action": "fold" if "fold" in legal else "check"}
            # 25-34% pot and not enough equity: fall through to the normal rules below

        # 1. Counter-Maniac Call-Down & Trap Engine
        if archetype == "MANIAC":
            if eq_adj >= 0.80 and inter >= 3 and "raise" in legal:
                target = opp_bet + int((pot + to_call) * 0.85)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                return {"type": "action", "action": "raise", "amount": amt}
            if street in ("turn", "river") and inter == 0 and eq_adj < 0.58:
                return {"type": "action", "action": "fold"}
            if (
                (inter >= 2 and eq_adj >= 0.40)
                or (inter >= 1 and eq_adj >= max(0.46, pot_odds - 0.03))
                or eq_adj >= pot_odds + 0.02
            ):
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        # 2. Facing a Postflop Raise Over Our Bet (opp_raised_our_bet) or Check-Raise:
        if opp_raised_our_bet or opp_check_raised:
            if (
                (inter == 4 and eq_adj >= 0.82 and (street != "river" or danger < 0.35))
                or (street != "river" and inter >= 3 and full_cat >= 3 and eq_adj >= 0.85 and danger < 0.45)
            ) and "raise" in legal:
                target = opp_bet + int((pot + to_call) * 0.75)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                return {"type": "action", "action": "raise", "amount": amt}

            if street in ("turn", "river") and inter <= 2:
                if inter == 2 and (pot_odds <= 0.22 or (freq_raiser and pot_odds <= 0.30)) and eq_adj >= 0.58:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

            if street in ("turn", "river") and inter == 3 and full_cat == 2:
                if (street == "river" and to_call >= 25 and (danger >= 0.15 or eq_adj < 0.78)) or eq_adj < 0.66 or (danger >= 0.50 and eq_adj < 0.72):
                    return {"type": "action", "action": "fold"}
                return {"type": "action", "action": "call"}

            if street in ("turn", "river") and inter == 3 and full_cat == 3:
                if eq_adj < 0.60 or (street == "river" and to_call >= 40 and danger >= 0.40 and eq_adj < 0.74):
                    return {"type": "action", "action": "fold"}
                return {"type": "action", "action": "call"}

            if street == "river" and full_cat == 4 and to_call >= 35 and danger >= 0.25 and eq_adj < 0.78:
                return {"type": "action", "action": "fold"}

            if street == "flop" and inter <= 1 and not has_draw and (eq_adj < 0.58 or to_call >= 12):
                return {"type": "action", "action": "fold"}

            if street == "flop" and inter == 2 and not has_draw and to_call >= 32 and bet_to_pot_ratio >= 0.72 and eq_adj < 0.76:
                return {"type": "action", "action": "fold"}

            eq_adj -= (0.02 if (street == "flop" and freq_raiser and (inter >= 2 or has_draw)) else 0.07)

        # 3. Structural Defense Against Flop, Turn & River Bets:
        if street == "flop":
            # With `has_strong_draw` filtering out 4-out single gutshots, fold unimproved High Card / gutshots
            # (`inter == 0, not has_draw`) to flop bets `>= 0.38x` pot or in 3-bet pots!
            if inter == 0 and not has_draw:
                if (bet_to_pot_ratio >= 0.38 or is_3bet_pot) and eq_adj < max(0.55, pot_odds + 0.12):
                    return {"type": "action", "action": "fold"}
            # Fold weak `inter == 1` hands (`eq_adj < 0.46`) only to big flop bets (`>= 0.64x` pot or 3-bet pots)
            # so our overall postflop fold rate stays below 42% (MDF) while avoiding 1.10x overbet traps (`L1 #105`)!
            if inter == 1 and not has_draw:
                if (bet_to_pot_ratio >= 0.64 or is_3bet_pot) and eq_adj < 0.46:
                    return {"type": "action", "action": "fold"}

        if street == "river":
            if inter == 0 and to_call > pot * 0.16:
                return {"type": "action", "action": "fold"}
            # v8 Fix: Against massive river overbet jams (`to_call >= 60` and `bet_to_pot_ratio >= 0.85`),
            # fold bare One Pair / Overpair (`inter <= 2`, e.g., `L2 #158`, `L2 #112`, `L1 #76`, `L2 #50`)!
            if to_call >= 60 and bet_to_pot_ratio >= 0.85 and inter <= 2 and eq_adj < 0.84 and self.opp.showdown_bluff_rate < 0.35:
                return {"type": "action", "action": "fold"}
            inter1_lim = 0.38 if (prev_checked_through or agg_postflop_opp) else 0.26
            inter1_eq_req = 0.56 if (prev_checked_through and agg_postflop_opp) else 0.66
            if inter == 1 and (to_call > pot * inter1_lim or (opp_was_prev_aggressor and not agg_postflop_opp)) and eq_adj < inter1_eq_req:
                return {"type": "action", "action": "fold"}
            if opp_bet_flop and opp_bet_turn and bet_to_pot_ratio >= 0.48 and self.opp.river_barrel_rate < 0.48:
                if inter <= 2 and eq_adj < 0.58:
                    return {"type": "action", "action": "fold"}
            if bet_to_pot_ratio >= 0.72 and self.opp.river_barrel_rate < 0.48:
                if inter <= 2 and eq_adj < 0.74:
                    return {"type": "action", "action": "fold"}
                if inter == 3 and full_cat == 2 and danger >= 0.40 and eq_adj < 0.75:
                    return {"type": "action", "action": "fold"}

        if street == "turn":
            if inter == 0 and not has_draw and bet_to_pot_ratio >= 0.28 and eq_adj < 0.58:
                return {"type": "action", "action": "fold"}
            if inter == 0 and has_draw and (bet_to_pot_ratio >= 0.50 or is_3bet_pot) and eq_adj < 0.54:
                return {"type": "action", "action": "fold"}
            if inter == 1 and not has_draw:
                if is_3bet_pot and bet_to_pot_ratio >= 0.42 and eq_adj < 0.62:
                    return {"type": "action", "action": "fold"}
                turn_inter1_eq = 0.52 if agg_postflop_opp else 0.63
                turn_ratio_lim = 0.68 if agg_postflop_opp else 0.55
                if (bet_to_pot_ratio >= turn_ratio_lim or (opp_was_prev_aggressor and not agg_postflop_opp)) and eq_adj < turn_inter1_eq:
                    return {"type": "action", "action": "fold"}
            if bet_to_pot_ratio >= 0.80 and (inter <= 1 or (inter == 2 and not agg_postflop_opp)) and eq_adj < 0.68:
                return {"type": "action", "action": "fold"}

        # 4. Raise for Value (when facing an opening bet, my_bet == 0)
        raise_val_thresh = 0.77
        if archetype == "CALLING_STATION":
            raise_val_thresh = 0.74
        elif archetype == "NIT_TAG":
            raise_val_thresh = 0.81

        is_pocket_set = (
            full_cat == 3
            and len(self.hole2) == 2
            and self.hole2[0][0] == self.hole2[1][0]
        )
        if street == "river":
            can_value_raise = (
                (full_cat >= 6 and inter >= 3)
                or (full_cat == 5 and inter == 4 and eq_adj >= 0.85)
                or (full_cat == 4 and inter == 4 and danger < 0.25 and eq_adj >= 0.85)
                or (is_pocket_set and danger < 0.30 and eq_adj >= 0.88)
            )
        elif street == "turn":
            can_value_raise = (
                (full_cat >= 4 and inter == 4 and danger < 0.45)
                or (is_pocket_set and danger < 0.45)
                or (full_cat == 3 and inter == 3 and eq_adj >= 0.88 and danger < 0.35)
                or (full_cat == 2 and inter == 3 and eq_adj >= 0.84 and danger < 0.25 and not is_3bet_pot)
            )
        else:
            can_value_raise = (full_cat >= 3 and inter >= 3) or (inter >= 3 and eq_adj >= 0.80 and danger < 0.45)

        if can_value_raise and eq_adj >= raise_val_thresh and "raise" in legal:
            mult = 0.85 if eq_adj > 0.84 else 0.65
            target = opp_bet + int((pot + to_call) * mult)
            amt = max(req["min_raise"], min(req["max_raise"], target))
            return {"type": "action", "action": "raise", "amount": amt}

        # 5. Standard Call Decision
        margin = 0.02
        if street == "flop":
            margin = -0.02 if self.seat == "button" else 0.0
        elif street == "river":
            margin = 0.03

        if archetype == "NIT_TAG" and to_call > pot * 0.40:
            margin += 0.05
        elif archetype == "CALLING_STATION" and to_call > pot * 0.45:
            margin += 0.08

        if eq_adj >= pot_odds + margin and "call" in legal:
            return {"type": "action", "action": "call"}

        return {"type": "action", "action": "fold" if "fold" in legal else "check"}


# ---------------------------------------------------------------------------
# Crash-proof main loop
# ---------------------------------------------------------------------------
_ERRORS = 0


def _log_error(where):
    global _ERRORS
    _ERRORS += 1
    if _ERRORS <= 20:
        sys.stderr.write(f"[bot error #{_ERRORS} in {where}]\n{traceback.format_exc()}\n")
        sys.stderr.flush()


def fallback_action(req):
    legal = req.get("legal_actions") or []
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal and req.get("to_call", 99) <= 2:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else "call"}


def legalize(reply, req):
    legal = req.get("legal_actions") or []
    if not isinstance(reply, dict) or reply.get("action") not in legal:
        return fallback_action(req)
    if reply["action"] == "raise":
        lo, hi = req.get("min_raise"), req.get("max_raise")
        if lo is None or hi is None:
            return fallback_action(req)
        try:
            amt = int(reply.get("amount", lo))
        except (TypeError, ValueError):
            amt = lo
        return {"type": "action", "action": "raise", "amount": max(lo, min(hi, amt))}
    return {"type": "action", "action": reply["action"]}


def fallback_discard(req):
    cards = req.get("hole_cards") or []
    return {"type": "discard", "card": min(cards, key=lambda c: RANK_VAL.get(c[0], 0)) if cards else ""}


def send(obj):
    try:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()
    except (BrokenPipeError, OSError):
        sys.exit(0)


def main():
    bot = AdaptiveBot()
    handlers = {
        "init": bot.handle_init,
        "new_hand": bot.handle_new_hand,
        "discard_result": bot.handle_discard_result,
        "hand_result": bot.handle_hand_result,
    }
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        if t == "action_request":
            try:
                reply = legalize(bot.choose_action(msg), msg)
            except Exception:
                _log_error("choose_action")
                reply = fallback_action(msg)
            send(reply)
        elif t == "discard_request":
            try:
                reply = bot.choose_discard(msg)
                if reply.get("card") not in msg.get("hole_cards", []):
                    reply = fallback_discard(msg)
            except Exception:
                _log_error("choose_discard")
                reply = fallback_discard(msg)
            send(reply)
        elif t == "match_end":
            break
        elif t in handlers:
            try:
                handlers[t](msg)
            except Exception:
                _log_error(t)


if __name__ == "__main__":
    main()
