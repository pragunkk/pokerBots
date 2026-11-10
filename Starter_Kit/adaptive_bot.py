"""
Adaptive Meta-Strategy Bot v6.1 (Updated for Official Two-Leg Match Format & Per-Leg Time Banks).
Key Updates for Organiser Rulebook & Engine Changes:
1. Two-Leg Duplicate Match Optimization (`LEGS = 2`, 200 hands/leg Stage 1, 500 hands/leg Stage 2):
   - Because each leg runs in an isolated fresh process and card luck only cancels across Leg 1 + Leg 2
     combined, unilateral single-leg score throttling (`PROTECT_WIN` / `COMEBACK`) is removed so the bot
     strictly maximizes expected chip EV on every hand of both legs!
2. Per-Leg Time Bank Calibration (`30,000 ms` Stage 1 / `75,000 ms` Stage 2):
   - Calibrates rollout thresholds to the new per-leg time banks (`> 8,000 ms` for full 95-rollout flop,
     150-rollout pre-evaluated turn, and 903-hand exact combinatorial river solver; `< 2,000 ms` emergency guard).
3. Full 903-Hand Exact Combinatorial River Solver (`exact_river_equity` in `fast_eval.py`)
4. Pre-Evaluated Turn Rollout Accelerator (`150` rollouts with cached `alt_flop_score`)
5. Sub-Threshold Value Sizing (`0.62x` pot for Strong Top Pair / Overpair) & Triple-Barrel One-Pair Defense
6. Extreme Heuristic & Boundary Exploiter (`Auto-Steal` vs Extreme Folders, `Anti-Shove-Spammer Shield`).
"""

import json
import random
import sys
from collections import deque
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

        self.postflop_bets_raises = 0
        self.postflop_calls_checks = 0
        self.faced_bet_postflop = 0
        self.folded_to_bet_postflop = 0
        self.raised_my_bet_postflop = 0
        self.check_raises = 0
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
        self.hand_saw_river = False
        self.hand_bet_river = False
        self.opp_checked_this_street = False
        self.me_bet_this_street = False
        self.me_opened_preflop = False
        self.opp_responded_to_open = False
        self.last_street_tracked = "preflop"
        self.processed_history_len = 0

    def start_hand(self):
        if self.hands_seen > 0:
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

            if street == "river":
                self.hand_saw_river = True

            if street != self.last_street_tracked:
                self.opp_checked_this_street = False
                self.me_bet_this_street = False
                self.last_street_tracked = street

            if actor == "opp":
                if street == "preflop":
                    if self.me_opened_preflop and not self.opp_responded_to_open:
                        self.opp_responded_to_open = True
                        self.faced_preflop_steal += 1
                        if action == "fold":
                            self.folded_to_preflop_steal += 1
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
                        if street == "river":
                            self.hand_bet_river = True
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
                    if preflop_raises_seen == 0:
                        self.me_opened_preflop = True
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
    def fold_to_steal(self):
        return (self.folded_to_preflop_steal + 2.0) / (self.faced_preflop_steal + 5.0)

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
    def postflop_agg_per_hand(self):
        g = (self.postflop_bets_raises + 3.0) / (self.hands_seen + 10.0)
        if len(self.recent_window) >= 18:
            r = sum(x[2] for x in self.recent_window) / len(self.recent_window)
            return 0.40 * g + 0.60 * r
        return g

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
            or three_bet_rate > 0.22
            or (self.pfr > 0.62 and self.postflop_agg_per_hand > 0.90)
            or (self.showdowns >= 4 and self.showdown_bluff_rate > 0.42 and self.postflop_agg_per_hand > 0.75)
        ):
            return "MANIAC"

        if self.check_raises >= 3 and self.check_raises / max(1, self.hands_seen) > 0.055:
            return "TRAPPER"

        if self.fold_to_bet < 0.28 and self.vpip > 0.58 and self.postflop_agg_per_hand < 0.45:
            return "CALLING_STATION"

        if self.vpip < 0.45 or self.fold_to_bet > 0.47 or self.fold_to_steal > 0.62:
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

    def handle_init(self, msg):
        self.total_hands = msg.get("total_hands", 200)
        self.starting_stack = msg.get("starting_stack", 200)
        self.time_bank_ms = msg.get("time_bank_ms", 30000)
        # In the updated 2-leg format, Stage 2 sends total_hands = 500 per leg (1000 match total)
        self.cap = 1000 if self.total_hands >= 500 else 400

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
        hist = msg.get("history", [])
        self.opp.observe_history(hist)
        if msg.get("result") == "showdown" and msg.get("opp_hole_cards"):
            opp_bet_late = any(
                h["actor"] == "opp" and h["street"] in ("turn", "river") and h["action"] == "raise"
                for h in hist
            )
            if opp_bet_late:
                self.opp.showdowns += 1
                inter = hole_interaction_level(msg["opp_hole_cards"], msg["board"])
                if inter <= 1:
                    self.opp.showdown_bluffs += 1

    def choose_discard(self, req):
        hole = req["hole_cards"]
        board = req["board"]
        if self.cached_flop_discard and set(hole) == set(self.hole3):
            best_card = self.cached_flop_discard
        else:
            time_left = req.get("time_left_ms", 30000)
            rollouts = 90 if time_left > 8000 else 35
            best_card, _, _ = evaluate_discard_options(
                hole, board, num_rollouts=rollouts, opp_vpip=self.opp.vpip
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
        if self.opp.hands_seen >= 8 and self.opp.fold_to_steal > 0.62 and opp_bet <= 2:
            if "raise" in legal:
                return {"type": "action", "action": "raise", "amount": req["min_raise"]}

        open_thresh = 0.32
        limp_thresh = 0.21
        call_raise_thresh = 0.43
        three_bet_val_thresh = 0.67

        if archetype == "NIT_TAG":
            open_thresh = 0.23
            call_raise_thresh = 0.48
            three_bet_val_thresh = 0.71
        elif archetype == "CALLING_STATION":
            open_thresh = 0.38
            limp_thresh = 0.22
        elif archetype == "MANIAC":
            open_thresh = 0.36
            call_raise_thresh = 0.36
            three_bet_val_thresh = 0.68

        if opp_bet <= 2:
            if strength >= open_thresh and "raise" in legal:
                if archetype == "CALLING_STATION" and strength > 0.60:
                    size = 7
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
            if strength >= 0.82 and "raise" in legal:
                target = min(req["max_raise"], max(req["min_raise"], opp_bet * 3))
                return {"type": "action", "action": "raise", "amount": target}
            if strength >= 0.49 or (self.seat == "button" and strength >= 0.45 and pot_odds <= 0.30):
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if strength >= three_bet_val_thresh and "raise" in legal:
            target = min(req["max_raise"], max(req["min_raise"], int(opp_bet * 2.8)))
            return {"type": "action", "action": "raise", "amount": target}

        if strength >= call_raise_thresh or (pot_odds < 0.28 and strength >= 0.34):
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

        if street == "flop":
            if self.cached_flop_eq is None:
                rollouts = 90 if time_left > 8000 else 40
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
            has_draw = has_strong_draw(keep2, board)
        else:
            rollouts = 145 if time_left > 8000 else 55
            if archetype == "MANIAC":
                opp_bias = -0.20 if to_call > 0 else 0.0
            elif archetype == "NIT_TAG" and to_call > 0:
                opp_bias = 0.25
            else:
                opp_bias = 0.16 if to_call > 0 else 0.0

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
            has_draw = has_strong_draw(self.hole2, board)

            if self.opp_discard and to_call > 0 and archetype not in ("MANIAC",):
                disc_r = RANK_VAL[self.opp_discard[0]]
                if disc_r >= 12:
                    eq *= 0.95

        danger = board_danger_level(board)
        pot_odds = to_call / max(1, pot + to_call)
        eq_adj = eq

        prev_street = {"flop": "preflop", "turn": "flop", "river": "turn"}[street]
        prev_acts = [h for h in hist if h["street"] == prev_street]
        prev_raises = [h for h in prev_acts if h["action"] == "raise"]
        opp_was_prev_aggressor = bool(prev_raises and prev_raises[-1]["actor"] == "opp")
        prev_checked_through = (street in ("turn", "river")) and len(prev_raises) == 0

        opp_bet_flop = any(h["street"] == "flop" and h["actor"] == "opp" and h["action"] == "raise" for h in hist)
        opp_bet_turn = any(h["street"] == "turn" and h["actor"] == "opp" and h["action"] == "raise" for h in hist)

        # ------------------------------------------------------------------
        # Scenario A: Checked to us (to_call == 0)
        # ------------------------------------------------------------------
        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}

            if self.opp.hands_seen >= 10 and self.opp.fold_to_bet > 0.68:
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.30)))
                return {"type": "action", "action": "raise", "amount": amt}

            # Rule 1: Do not "donk-bet" into the previous street's raiser on Turn/River without Two Pair+ (`inter >= 3`)
            if street in ("turn", "river") and opp_was_prev_aggressor and inter < 3 and eq_adj < 0.78:
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

            # Rule 3: Pot-control weak/underpairs (`inter <= 1`) on Turn & River!
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
            elif archetype == "TRAPPER":
                val_thresh = 0.62

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
            probe_frac = 0.38
            if street == "flop" and 0.38 <= eq_adj < val_thresh and archetype not in ("CALLING_STATION", "MANIAC"):
                bluff_freq = 0.52 if archetype == "NIT_TAG" else 0.36
                probe_frac = 0.36
            elif (
                street == "turn"
                and prev_checked_through
                and 0.30 <= eq_adj < val_thresh
                and archetype not in ("CALLING_STATION", "MANIAC")
            ):
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

        # 2. Facing a Postflop Raise Over Our Bet (opp_raised_our_bet) or Check-Raise against a Non-Maniac:
        if opp_raised_our_bet or opp_check_raised:
            if (
                (inter == 4 and eq_adj >= 0.80)
                or (street != "river" and inter >= 3 and eq_adj >= 0.85 and danger < 0.45)
            ) and "raise" in legal:
                target = opp_bet + int((pot + to_call) * 0.75)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                return {"type": "action", "action": "raise", "amount": amt}

            if street in ("turn", "river") and inter <= 2:
                if inter == 2 and pot_odds <= 0.20 and eq_adj >= 0.62:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

            if street in ("turn", "river") and inter == 3 and full_cat == 2:
                if eq_adj < 0.66 or (danger >= 0.50 and eq_adj < 0.72):
                    return {"type": "action", "action": "fold"}
                return {"type": "action", "action": "call"}

            if street in ("turn", "river") and inter == 3 and full_cat == 3:
                if eq_adj < 0.60:
                    return {"type": "action", "action": "fold"}
                return {"type": "action", "action": "call"}

            if street == "flop" and inter <= 1 and not has_draw and eq_adj < 0.60:
                return {"type": "action", "action": "fold"}

            eq_adj -= 0.07

        # 3. Structural Defense Against Big Turn & River Bets / Triple Barrels:
        if street == "river":
            if inter == 0 and to_call > pot * 0.16:
                return {"type": "action", "action": "fold"}
            inter1_lim = 0.33 if prev_checked_through else 0.26
            if inter == 1 and (to_call > pot * inter1_lim or opp_was_prev_aggressor) and eq_adj < 0.68:
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
            if inter == 1 and (bet_to_pot_ratio >= 0.55 or opp_was_prev_aggressor) and eq_adj < 0.64:
                return {"type": "action", "action": "fold"}
            if bet_to_pot_ratio >= 0.80 and inter <= 2 and eq_adj < 0.68:
                return {"type": "action", "action": "fold"}

        # 4. Raise for Value (when facing an opening bet, my_bet == 0)
        raise_val_thresh = 0.77
        if archetype == "CALLING_STATION":
            raise_val_thresh = 0.74
        elif archetype == "NIT_TAG":
            raise_val_thresh = 0.81

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
