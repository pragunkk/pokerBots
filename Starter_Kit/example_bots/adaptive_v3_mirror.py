import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Adaptive Meta-Strategy Bot v3 for Discard Hold'em (Checkpoints 3 & 4 Championship Engine).
Key Architectural Features:
1. O(1) Precomputed 1,755 Canonical 3-Card Preflop Equity Lookup (`PREFLOP_TABLE_1755`)
2. Dual-Window Opponent Profiler (Global + 50-Hand Sliding Window):
   - Strictly separates true MANIAC (high VPIP + high PFR + high postflop barrels per hand + low fold rate)
     from Tight-Aggressive (TAG) and GTO/Balanced bots.
   - Tracks `postflop_raise_vs_bet_rate` to know whether an opponent's raise over our bet is a value-nutted
     signal (TAG/GTO/Station) or a high-frequency bluff (Maniac).
3. Structural Hand Category & Board-Relative Gating (`hole_interaction_level`):
   - Prevents calling multi-street big bets with middle/bottom pair or underpairs (`inter == 1`).
   - Prevents postflop 3-betting / jamming with mere One Pair (`cat == 1`) into Two Pair/Sets/Straights.
   - Prevents "donk-betting" into the previous street's raiser without a monster hand.
4. Deceptive Discard Tie-Breaking & Bayesian Opponent Discard Reader:
   - Reads opponent's exposed flop discard (`opp_discard`) for card removal and range narrowing.
5. Match-Point & Capped-Score Variance Manager (`PROTECT_WIN` / `COMEBACK` / `NORMAL`).
"""

import json
import random
import sys
from collections import deque
from fast_eval import (
    RANK_VAL,
    eval_cards,
    evaluate_discard_options,
    hole_interaction_level,
    post_discard_equity,
    preflop_3card_strength,
)


class OpponentProfile:
    def __init__(self, window_size=50):
        self.hands_seen = 0
        self.vpip_hands = 0
        self.pfr_hands = 0
        self.three_bets = 0

        self.postflop_bets_raises = 0
        self.postflop_calls_checks = 0
        self.faced_bet_postflop = 0
        self.folded_to_bet_postflop = 0
        self.raised_my_bet_postflop = 0
        self.check_raises = 0

        self.high_card_discards = 0
        self.total_discards = 0
        self.showdowns = 0
        self.showdown_bluffs = 0

        # Sliding window of recent hands: (vpip, pfr, post_agg_acts, post_pass_acts, folded_to_bet, faced_bet, raised_my_bet)
        self.recent_window = deque(maxlen=window_size)
        self.reset_hand_flags()

    def reset_hand_flags(self):
        self.hand_vpip = 0
        self.hand_pfr = 0
        self.hand_agg = 0
        self.hand_pass = 0
        self.hand_folded = 0
        self.hand_faced = 0
        self.hand_raised_me = 0
        self.opp_checked_this_street = False
        self.me_bet_this_street = False
        self.last_street_tracked = "preflop"
        self.processed_history_len = 0

    def start_hand(self):
        if self.hands_seen > 0:
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

    def observe_history(self, history):
        new_entries = history[self.processed_history_len:]
        self.processed_history_len = len(history)

        preflop_raises_seen = 0
        for entry in history[: self.processed_history_len - len(new_entries)]:
            if entry["street"] == "preflop" and entry["action"] == "raise":
                preflop_raises_seen += 1

        for entry in new_entries:
            street = entry["street"]
            actor = entry["actor"]
            action = entry["action"]

            if street != self.last_street_tracked:
                self.opp_checked_this_street = False
                self.me_bet_this_street = False
                self.last_street_tracked = street

            if actor == "opp":
                if street == "preflop":
                    if action in ("call", "raise") and not self.hand_vpip:
                        self.hand_vpip = 1
                        self.vpip_hands += 1
                    if action == "raise":
                        if not self.hand_pfr:
                            self.hand_pfr = 1
                            self.pfr_hands += 1
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
                        if self.me_bet_this_street:
                            self.raised_my_bet_postflop += 1
                            self.hand_raised_me += 1
                        if self.opp_checked_this_street:
                            self.check_raises += 1
                    elif action == "fold":
                        self.folded_to_bet_postflop += 1
                        self.hand_folded += 1

            elif actor == "me":
                if street == "preflop" and action == "raise":
                    preflop_raises_seen += 1
                elif street != "preflop" and action == "raise":
                    self.faced_bet_postflop += 1
                    self.hand_faced += 1
                    self.me_bet_this_street = True

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
    def fold_to_bet(self):
        g = (self.folded_to_bet_postflop + 4.0) / (self.faced_bet_postflop + 10.0)
        if len(self.recent_window) >= 18:
            r_faced = sum(x[5] for x in self.recent_window)
            if r_faced >= 6:
                r = sum(x[4] for x in self.recent_window) / r_faced
                return 0.40 * g + 0.60 * r
        return g

    @property
    def postflop_agg_per_hand(self):
        g = (self.postflop_bets_raises + 3.0) / (self.hands_seen + 10.0)
        if len(self.recent_window) >= 18:
            r = sum(x[2] for x in self.recent_window) / len(self.recent_window)
            return 0.40 * g + 0.60 * r
        return g

    @property
    def raise_my_bet_rate(self):
        return (self.raised_my_bet_postflop + 1.0) / (self.faced_bet_postflop + 10.0)

    @property
    def showdown_bluff_rate(self):
        return (self.showdown_bluffs + 0.5) / (self.showdowns + 4.0)

    def classify(self):
        """
        Precisely classifies the opponent into:
          - 'MANIAC': Hyper-aggressive preflop & postflop (e.g. lag_maniac_bot)
          - 'CALLING_STATION': Loose-passive, rarely folds or raises (e.g. calling_station_bot, starter_bot)
          - 'TRAPPER': Frequent check-raiser (e.g. trapper_bot)
          - 'NIT_TAG': Tight or high fold-to-bet (e.g. tag_bot)
          - 'BALANCED': Solid/GTO or adaptive opponent (e.g. cfr_gto_bot, rival_adaptive_bot)
        """
        if self.hands_seen < 12:
            return "BALANCED"

        # True Maniac: raises > 52% preflop AND fires > 0.85 postflop bets/raises per hand (or 3-bets > 18%)
        three_bet_rate = self.three_bets / max(1, self.hands_seen)
        if (self.pfr > 0.54 and self.postflop_agg_per_hand > 0.80) or three_bet_rate > 0.18 or self.postflop_agg_per_hand > 1.05:
            return "MANIAC"

        if self.check_raises >= 3 and self.check_raises / max(1, self.hands_seen) > 0.055:
            return "TRAPPER"

        if self.fold_to_bet < 0.28 and self.vpip > 0.58 and self.postflop_agg_per_hand < 0.45:
            return "CALLING_STATION"

        if self.vpip < 0.46 or self.fold_to_bet > 0.48:
            return "NIT_TAG"

        return "BALANCED"


class AdaptiveBot:
    def __init__(self):
        self.total_hands = 400
        self.starting_stack = 200
        self.time_bank_ms = 60000
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

    def handle_init(self, msg):
        self.total_hands = msg.get("total_hands", 400)
        self.starting_stack = msg.get("starting_stack", 200)
        self.time_bank_ms = msg.get("time_bank_ms", 60000)
        self.cap = 1000 if self.total_hands > 500 else 400

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
        self.opp.start_hand()

    def handle_discard_result(self, msg):
        self.my_discard = msg["my_discard"]
        self.opp_discard = msg["opp_discard"]
        self.hole2 = list(msg["hole_cards"])
        self.opp.observe_discard(self.opp_discard)

    def handle_hand_result(self, msg):
        self.cum_net += msg.get("net", 0)
        self.opp.observe_history(msg.get("history", []))
        if msg.get("result") == "showdown" and msg.get("opp_hole_cards"):
            self.opp.showdowns += 1
            inter = hole_interaction_level(msg["opp_hole_cards"], msg["board"])
            if inter <= 1:
                self.opp.showdown_bluffs += 1

    def get_risk_mode(self):
        hands_left = max(1, self.total_hands - self.hand_id + 1)
        if self.cum_net > self.cap + 140:
            return "PROTECT_WIN"
        if hands_left <= 45 and self.cum_net > max(85, hands_left * 3.5):
            return "PROTECT_WIN"
        if hands_left <= 45 and self.cum_net < -20 and self.cum_net > -hands_left * 15:
            return "COMEBACK"
        return "NORMAL"

    def choose_discard(self, req):
        hole = req["hole_cards"]
        board = req["board"]
        if self.cached_flop_discard and set(hole) == set(self.hole3):
            best_card = self.cached_flop_discard
            best_eq = self.cached_flop_eq
            eq_map = self.cached_flop_eq_map
        else:
            time_left = req.get("time_left_ms", 30000)
            rollouts = 75 if time_left > 12000 else 35
            best_card, best_eq, eq_map = evaluate_discard_options(
                hole, board, num_rollouts=rollouts, opp_vpip=self.opp.vpip
            )

        close_cards = [c for c, eq in eq_map.items() if best_eq - eq <= 0.008]
        if len(close_cards) > 1:
            if best_eq > 0.72:
                best_card = min(close_cards, key=lambda c: RANK_VAL[c[0]])
            else:
                best_card = max(close_cards, key=lambda c: RANK_VAL[c[0]])

        return {"type": "discard", "card": best_card}

    def choose_action(self, req):
        self.opp.observe_history(req.get("history", []))
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        time_left = req.get("time_left_ms", 30000)

        archetype = self.opp.classify()
        risk_mode = self.get_risk_mode()

        if time_left < 2500:
            if "check" in legal:
                return {"type": "action", "action": "check"}
            if to_call <= 4 and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if street == "preflop":
            return self._play_preflop(req, archetype, risk_mode)
        else:
            return self._play_postflop(req, archetype, risk_mode, time_left)

    def _play_preflop(self, req, archetype, risk_mode):
        legal = req["legal_actions"]
        to_call = req["to_call"]
        opp_bet = req["opp_bet"]
        my_bet = req["my_bet"]
        strength = preflop_3card_strength(self.hole3)

        open_thresh = 0.34
        limp_thresh = 0.22
        call_raise_thresh = 0.44
        three_bet_val_thresh = 0.68

        if archetype == "NIT_TAG":
            open_thresh = 0.25
            call_raise_thresh = 0.48
            three_bet_val_thresh = 0.72
        elif archetype == "CALLING_STATION":
            open_thresh = 0.40
            limp_thresh = 0.24
        elif archetype == "MANIAC":
            open_thresh = 0.36
            call_raise_thresh = 0.36
            three_bet_val_thresh = 0.68

        if risk_mode == "PROTECT_WIN":
            open_thresh += 0.08
            call_raise_thresh += 0.08
            three_bet_val_thresh += 0.10
        elif risk_mode == "COMEBACK":
            open_thresh -= 0.05
            three_bet_val_thresh -= 0.05

        if opp_bet <= 2:
            if strength >= open_thresh and "raise" in legal:
                size = 7 if (archetype == "CALLING_STATION" and strength > 0.60) else 6
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
            if risk_mode == "PROTECT_WIN":
                req_s = max(req_s, 0.82)
            if strength >= req_s:
                if strength >= 0.85 and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": req["max_raise"]}
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if archetype == "MANIAC" and my_bet >= 6:
            if strength >= 0.74 and "raise" in legal:
                target = min(req["max_raise"], max(req["min_raise"], opp_bet * 3))
                return {"type": "action", "action": "raise", "amount": target}
            if strength >= 0.40:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        # If we already raised preflop and a NON-MANIAC 3-bets us (my_bet >= 6, opp_bet >= 16):
        # Respect their 3-bet range! Only 4-bet with absolute monsters (0.82+) and call with strong playable hands (0.52+)
        if my_bet >= 6 and opp_bet >= 14:
            if strength >= 0.82 and "raise" in legal:
                target = min(req["max_raise"], max(req["min_raise"], opp_bet * 3))
                return {"type": "action", "action": "raise", "amount": target}
            if strength >= 0.52:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if strength >= three_bet_val_thresh and "raise" in legal:
            target = min(req["max_raise"], max(req["min_raise"], opp_bet * 3))
            return {"type": "action", "action": "raise", "amount": target}

        if strength >= call_raise_thresh or (pot_odds < 0.27 and strength >= 0.35):
            return {"type": "action", "action": "call"}

        return {"type": "action", "action": "fold"}

    def _play_postflop(self, req, archetype, risk_mode, time_left):
        legal = req["legal_actions"]
        street = req["street"]
        board = req["board"]
        to_call = req["to_call"]
        pot = req["pot"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        hist = req.get("history", [])

        # Calculate equity and structural board-relative interaction level
        if street == "flop":
            if self.cached_flop_eq is None:
                rollouts = 75 if time_left > 15000 else 40
                best_c, best_eq, eq_map = evaluate_discard_options(
                    self.hole3, board, num_rollouts=rollouts, opp_vpip=self.opp.vpip
                )
                self.cached_flop_discard = best_c
                self.cached_flop_eq = best_eq
                self.cached_flop_eq_map = eq_map
            eq = self.cached_flop_eq
            keep2 = [c for c in self.hole3 if c != self.cached_flop_discard][:2]
            inter = hole_interaction_level(keep2, board)
            full_cat = eval_cards(keep2, board) >> 20
        else:
            rollouts = 95 if time_left > 15000 else 45
            if archetype == "MANIAC":
                opp_bias = -0.20 if to_call > 0 else 0.0
            elif archetype == "NIT_TAG" and to_call > 0:
                opp_bias = 0.25
            else:
                opp_bias = 0.15 if to_call > 0 else 0.0

            eq = post_discard_equity(
                self.hole2,
                board,
                self.my_discard,
                self.opp_discard,
                opp_range_bias=opp_bias,
                num_rollouts=rollouts,
            )
            inter = hole_interaction_level(self.hole2, board)
            full_cat = eval_cards(self.hole2, board) >> 20

            if self.opp_discard and to_call > 0 and archetype not in ("MANIAC",):
                disc_r = RANK_VAL[self.opp_discard[0]]
                if disc_r >= 12:
                    eq *= 0.95

        pot_odds = to_call / max(1, pot + to_call)

        if risk_mode == "PROTECT_WIN":
            eq_adj = eq - 0.05
        elif risk_mode == "COMEBACK":
            eq_adj = eq + 0.04
        else:
            eq_adj = eq

        # Check who was the aggressor on the previous street
        prev_street = {"flop": "preflop", "turn": "flop", "river": "turn"}[street]
        prev_raises = [h for h in hist if h["street"] == prev_street and h["action"] == "raise"]
        opp_was_prev_aggressor = bool(prev_raises and prev_raises[-1]["actor"] == "opp")

        # ------------------------------------------------------------------
        # Scenario A: Checked to us (to_call == 0)
        # ------------------------------------------------------------------
        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}

            # Rule 1: Do not "donk-bet" into the previous street's raiser on Turn/River without Two Pair+ (inter >= 3)
            if street in ("turn", "river") and opp_was_prev_aggressor and inter < 3 and eq_adj < 0.76:
                return {"type": "action", "action": "check"}

            # Rule 2: Against a MANIAC, check-call medium hands on Flop/Turn so they bluff into us!
            if archetype == "MANIAC":
                if street in ("flop", "turn") and 0.48 <= eq_adj <= 0.75:
                    return {"type": "action", "action": "check"}
                if eq_adj > 0.75 or (street == "river" and eq_adj >= 0.58 and inter >= 2):
                    frac = 0.80 if eq_adj > 0.80 else 0.60
                    target = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                    return {"type": "action", "action": "raise", "amount": target}
                return {"type": "action", "action": "check"}

            # Rule 3: On Turn & River, do not value-bet weak/middle pair (`inter <= 1`) unless equity is huge!
            # Checking behind / pot-controlling weak pairs gets us to a cheap showdown.
            if street == "river" and inter <= 1 and eq_adj < 0.72:
                # Only bluff if we have terrible showdown value (inter == 0, eq_adj < 0.25) against a folder!
                if inter == 0 and eq_adj < 0.25 and (archetype == "NIT_TAG" or self.opp.fold_to_bet > 0.48):
                    if risk_mode != "PROTECT_WIN" and random.random() < 0.30:
                        target = max(req["min_raise"], min(req["max_raise"], int(pot * 0.55)))
                        return {"type": "action", "action": "raise", "amount": target}
                return {"type": "action", "action": "check"}

            if street == "turn" and inter == 1 and eq_adj < 0.64:
                return {"type": "action", "action": "check"}

            val_thresh = 0.57
            if archetype == "CALLING_STATION":
                val_thresh = 0.56 if street == "flop" else 0.60
            elif archetype == "TRAPPER":
                val_thresh = 0.63

            if eq_adj >= val_thresh:
                if archetype == "CALLING_STATION":
                    frac = 0.80 if eq_adj > 0.74 else 0.62
                elif eq_adj > 0.82:
                    frac = 0.78 if street == "river" else 0.68
                else:
                    frac = 0.55
                target = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                return {"type": "action", "action": "raise", "amount": target}

            # Semi-Bluff / Continuation Bet Probe
            bluff_freq = 0.0
            if archetype == "NIT_TAG" or self.opp.fold_to_bet > 0.48:
                if street == "flop" and 0.36 <= eq_adj < val_thresh:
                    bluff_freq = 0.50
                elif street == "turn" and 0.36 <= eq_adj < val_thresh and not opp_was_prev_aggressor:
                    bluff_freq = 0.38
            elif archetype == "BALANCED":
                if street == "flop" and 0.42 <= eq_adj < val_thresh:
                    bluff_freq = 0.30

            if risk_mode == "PROTECT_WIN":
                bluff_freq *= 0.25

            if bluff_freq > 0 and random.random() < bluff_freq:
                target = max(req["min_raise"], min(req["max_raise"], int(pot * 0.52)))
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

        # 1. Counter-Maniac Call-Down & Trap Engine
        if archetype == "MANIAC":
            if eq_adj >= 0.80 and inter >= 2 and "raise" in legal:
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

        # 2. Facing a Postflop Raise Over Our Bet (opp_raised_our_bet) or Check-Raise against a Non-Maniac:
        # In Discard Hold'em, when a non-Maniac raises our postflop bet, they almost ALWAYS hold Two Pair,
        # Set, Straight, or Flush (or monster combo draw on the flop)!
        if opp_raised_our_bet or opp_check_raised:
            # Never re-raise (3-bet the street) unless we have Two Pair+ (inter >= 3) and eq_adj >= 0.82!
            if inter >= 3 and eq_adj >= 0.82 and "raise" in legal:
                target = opp_bet + int((pot + to_call) * 0.75)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                return {"type": "action", "action": "raise", "amount": amt}

            # On Turn or River, if we only have One Pair or worse (inter <= 2, full_cat <= 1),
            # fold to a raise unless pot odds are tiny (< 0.22) and we have Top Pair/Overpair (inter == 2)!
            if street in ("turn", "river") and inter <= 2 and full_cat <= 1:
                if inter == 2 and pot_odds <= 0.22 and eq_adj >= 0.62:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

            # On the Flop, if we only have middle/bottom pair (inter <= 1) without a huge draw, fold to a raise
            if street == "flop" and inter <= 1 and eq_adj < 0.60:
                return {"type": "action", "action": "fold"}

            eq_adj -= 0.07

        # 3. Structural Gating Against Big Turn/River Bets (even if not a raise over our bet):
        # - Never call a meaningful River bet (> 18% pot) with High Card / Board-Only Pair (inter == 0)
        if street == "river" and inter == 0 and to_call > pot * 0.18:
            return {"type": "action", "action": "fold"}

        # - Do not call big River bets (> 45% pot) or Multi-Street Turn+River barrels with Underpair/Weak Pair (inter == 1)
        if street == "river" and inter == 1 and (to_call > pot * 0.30 or opp_was_prev_aggressor) and eq_adj < 0.66:
            return {"type": "action", "action": "fold"}
        if street == "turn" and inter == 1 and to_call > pot * 0.38 and opp_was_prev_aggressor and eq_adj < 0.62:
            return {"type": "action", "action": "fold"}

        # 4. Raise for Value (when facing an opening bet, my_bet == 0)
        # Only raise a postflop bet for value when holding Two Pair+ (inter >= 3) or monster Top Pair/Draw on Flop!
        raise_val_thresh = 0.76
        if archetype == "CALLING_STATION":
            raise_val_thresh = 0.74
        elif archetype == "NIT_TAG":
            raise_val_thresh = 0.80

        can_value_raise = (inter >= 3) or (street == "flop" and inter >= 2 and eq_adj >= 0.80)
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


def main():
    bot = AdaptiveBot()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        t = msg.get("type")
        if t == "init":
            bot.handle_init(msg)
        elif t == "new_hand":
            bot.handle_new_hand(msg)
        elif t == "discard_result":
            bot.handle_discard_result(msg)
        elif t == "hand_result":
            bot.handle_hand_result(msg)
        elif t == "action_request":
            reply = bot.choose_action(msg)
            print(json.dumps(reply), flush=True)
        elif t == "discard_request":
            reply = bot.choose_discard(msg)
            print(json.dumps(reply), flush=True)
        elif t == "match_end":
            break


if __name__ == "__main__":
    main()
