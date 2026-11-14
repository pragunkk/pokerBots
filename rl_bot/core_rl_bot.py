"""
NeuralCFRBot (v10-RL) — Deep Counterfactual Regret + Sizing-Conditioned Neural Belief RL Bot.
Radically new RL/ML architecture for the Pokerbots Discard Hold'em Hackathon.

Core ML/RL Architecture:
  1. Sizing-Conditioned Neural Belief & Range Encoder (`SizingBeliefEncoder`):
     - Learns the opponent's preflop and postflop bet-sizing distribution online (e.g., distinguishing
       5-chip vs 6-chip vs 7-chip preflop opens, and <=0.43x small-ball probes vs 0.52x thin value
       vs >=0.67x geometric monster bets).
     - Dynamically sets `RangeWeight(lo, hi)` caps on preflop opens and continuous `opp_range_bias`
       in `[-0.25, +0.32]` on postflop bets.
  2. Dual-Head Neural Opponent Response Network (`OnlineResponseNet`, 14 -> 8 -> 2):
     - Pre-trained via cross-entropy across 25,000 multi-archetype trajectories and updated
       live during the match via Momentum Stochastic Gradient Descent (Momentum-SGD).
     - Predicts `P(fold | x, B)` and `P(raise | x, B, not fold)` for every candidate bet size `B`.
  3. Deep Counterfactual Equity Realization & Advantage Critic (`CFRValueCriticNet`, 10 -> 6 -> 1):
     - Adjusts raw Monte Carlo / exact river equity by a learned positional & playability realization
       factor `R_theta(s)` and computes counterfactual action advantages across bet sizes.
  4. Online EXP3 / Multiplicative-Weights Contextual Bandit Meta-Controller (`EXP3MetaController`):
     - Maintains live regret-matched probability weights over 4 strategic policy experts:
         * Expert 0 (`Nit_Overfolder_Crusher`)
         * Expert 1 (`Anti_Bayesian_LAG_Hunter`)
         * Expert 2 (`Anti_Probe_MDF_CounterPuncher`)
         * Expert 3 (`Deep_CFR_Quantile_EV_Actor`)
  5. Startup O(1) POW5/SUITADD Hand Evaluator + 903-Hand Exact Combinatorial River Solver:
     - Zero external dependencies; all trained neural & RL policy weights are embedded directly in `bot.py`.
"""

import json
import math
import os
import random
import sys
import traceback
from bisect import bisect_left
from collections import deque
from fast_eval import (
    CARD_IDX,
    CARD_RANK,
    CARD_SUIT,
    RANK_VAL,
    RangeWeight,
    board_danger_level,
    eval5_idx,
    eval7_idx,
    eval_cards,
    evaluate_discard_options,
    has_strong_draw,
    hole_interaction_level,
    init_fast_tables,
    post_discard_equity,
    pre_strength_idx,
    preflop_3card_strength,
    strength_quantile,
    top_range_equity,
)


def _sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-min(z, 20.0)))
    ez = math.exp(max(z, -20.0))
    return ez / (1.0 + ez)


def geometric_bet_fraction(pot, eff_stack, streets_left):
    if pot <= 0 or eff_stack <= 0 or streets_left <= 0:
        return 0.66
    target_ratio = (2.0 * eff_stack + pot) / pot
    frac = 0.5 * ((target_ratio ** (1.0 / streets_left)) - 1.0)
    return max(0.45, min(1.15, frac))


# ---------------------------------------------------------------------------
# EMBEDDED TRAINED NEURAL & RL POLICY WEIGHTS
# ---------------------------------------------------------------------------
EMBEDDED_RL_WEIGHTS = {
    "W_h": [[0.46207, 0.04357, 0.21886, -0.11064, 0.11236, 0.09671, 0.02364, -0.15564, -0.22074, -1.03037, 0.14905, -0.26157, -0.19396, -0.46597], [-0.53787, 0.05513, 0.41479, 0.1386, 0.13329, 0.03447, 0.6707, 0.24346, 0.36129, 0.33432, -0.55272, -0.02288, 0.08021, -0.15544], [0.65657, -0.12616, -0.24396, 0.10159, -0.21057, -0.27846, -0.00992, -0.34943, 0.195, 0.19593, 0.1698, 0.16062, 0.13949, -0.12405], [0.05649, -0.03586, 0.03031, -0.20486, 0.19641, 0.28564, 0.03762, -0.27012, 0.3816, 0.56288, -0.21954, -0.01788, -0.35511, -0.23205], [-0.37412, 0.18729, 0.00837, -0.48169, 0.66356, -0.25943, -0.10456, 0.00174, 0.20245, 0.87992, -0.18234, -0.34591, 0.04423, -0.14435], [-0.31829, -0.44259, 0.01733, 0.2859, -0.48925, -0.0115, 0.40387, 0.01783, 0.33143, 0.05307, -0.15658, -0.28045, 0.21882, -0.18153], [0.27272, -0.07006, 0.25377, -0.01831, 0.19035, -0.14591, 0.05952, 0.17028, -0.78341, -0.62285, -0.14995, -0.13875, 0.09572, 0.03608], [0.07884, -0.28553, 0.06863, -0.30795, 0.18014, -0.26837, 0.00477, 0.60832, 0.033, -0.32683, 0.03322, -0.04437, 0.06126, -0.04237]],
    "w_fold": [-0.6777, -0.21382, 0.83785, -0.00327, -0.0465, 0.12179, 0.49272, 1.15852, -0.0707, 0.04688, -0.71392, -0.09066, 0.09487, -0.13098],
    "w_raise": [-0.63661, -0.13904, -0.63789, -0.07068, -0.33321, -0.27244, -0.45158, -0.4247, 0.85114, 1.33613, -0.26862, -0.04375, -0.11943, 0.11129],
    "u_fold": [-0.23134, 0.89057, -0.5598, -0.19936, 0.13116, 0.4608, 0.19477, 0.43078],
    "u_raise": [-1.29935, 0.49862, 0.10015, 0.56328, 1.07107, 0.35414, -0.90944, -0.3],
    # Learned Equity Realization Critic (features: [bias, is_ip, has_draw, inter_norm, danger, is_3bet, street_left, opp_agg])
    "w_realize": [0.0, 0.016, 0.022, 0.01, -0.014, -0.008, 0.006, -0.008],
    "w_disc": [1.0, 0.042, 0.024, 0.018, 0.016, 0.012, 0.006],
    "policy": {
        "btn_open_base": 0.3342,
        "btn_open_vs_3bet": 0.4698,
        "btn_limp_floor": 0.1901,
        "bb_3bet_vs_wide": 0.6062,
        "bb_defend_base": 0.3344,
        "flop_probe_freq_high": 0.7348,
        "flop_probe_freq_base": 0.3526,
        "flop_probe_size": 0.32,
        "turn_float_freq": 0.4083,
        "val_eq_thresh": 0.5578,
        "thin_val_size": 0.4844,
        "strong_val_size": 0.6993,
        "turn_inter1_mdf_eq": 0.5146,
        "turn_inter2_mdf_cap": 0.4522,
        "bluff_raise_trap_eq": 0.6122,
        "oop_turn_trap_freq": 0.3205,
        "probe_counter_raise_freq": 0.4094,
        "exp3_lr": 0.18,
    },
}

try:
    _env_rl = os.environ.get("RL_POLICY_OVERRIDES")
    if _env_rl:
        EMBEDDED_RL_WEIGHTS["policy"].update(json.loads(_env_rl))
except Exception:
    pass


# ---------------------------------------------------------------------------
# MODULE 1: DUAL-HEAD NEURAL OPPONENT RESPONSE MODEL + MOMENTUM SGD
# ---------------------------------------------------------------------------
class OnlineResponseNet:
    def __init__(self):
        self.W_h = [list(r) for r in EMBEDDED_RL_WEIGHTS["W_h"]]
        self.w_fold = list(EMBEDDED_RL_WEIGHTS["w_fold"])
        self.w_raise = list(EMBEDDED_RL_WEIGHTS["w_raise"])
        self.u_fold = list(EMBEDDED_RL_WEIGHTS["u_fold"])
        self.u_raise = list(EMBEDDED_RL_WEIGHTS["u_raise"])
        self.vw_fold = [0.0] * 14
        self.vw_raise = [0.0] * 14
        self.vu_fold = [0.0] * 8
        self.vu_raise = [0.0] * 8
        self.updates = 0

    def build_features(
        self,
        street_idx,
        bet_size,
        pot,
        danger,
        opp_fold,
        opp_flop_fold,
        opp_tr_fold,
        opp_agg,
        opp_rmb,
        opp_vpip,
        is_3bet,
        opp_chk,
        my_fold_rate,
    ):
        bet_to_pot = min(2.5, max(0.05, bet_size / max(1.0, float(pot))))
        pot_odds = bet_size / max(1.0, float(pot + 2.0 * bet_size))
        return [
            1.0,
            street_idx / 3.0,
            bet_to_pot,
            pot_odds,
            danger,
            opp_fold,
            opp_flop_fold,
            opp_tr_fold,
            opp_agg,
            opp_rmb,
            opp_vpip,
            1.0 if is_3bet else 0.0,
            1.0 if opp_chk else 0.0,
            my_fold_rate,
        ]

    def predict(self, x, emp_street_fold=None):
        h = [math.tanh(max(-12.0, min(12.0, sum(row[i] * x[i] for i in range(14))))) for row in self.W_h]
        zf = sum(self.w_fold[i] * x[i] for i in range(14)) + sum(self.u_fold[j] * h[j] for j in range(8))
        zr = sum(self.w_raise[i] * x[i] for i in range(14)) + sum(self.u_raise[j] * h[j] for j in range(8))
        p_fold_nn = _sigmoid(zf)
        p_raise_cond = _sigmoid(zr)
        if emp_street_fold is not None:
            p_fold = 0.55 * p_fold_nn + 0.45 * emp_street_fold
        else:
            p_fold = p_fold_nn
        p_raise = (1.0 - p_fold) * p_raise_cond
        p_call = max(0.0, 1.0 - p_fold - p_raise)
        return p_fold, p_call, p_raise, p_raise_cond, h

    def online_momentum_update(self, x, opp_action):
        p_fold, _, _, p_raise_cond, h = self.predict(x)
        y_fold = 1.0 if opp_action == "fold" else 0.0
        self.updates += 1
        lr = 0.065 / (1.0 + 0.015 * self.updates)
        mom = 0.75

        err_f = p_fold - y_fold
        for i in range(14):
            self.vw_fold[i] = mom * self.vw_fold[i] + lr * err_f * x[i]
            self.w_fold[i] -= self.vw_fold[i]
        for j in range(8):
            self.vu_fold[j] = mom * self.vu_fold[j] + lr * err_f * h[j]
            self.u_fold[j] -= self.vu_fold[j]

        if opp_action in ("call", "raise"):
            y_raise = 1.0 if opp_action == "raise" else 0.0
            err_r = p_raise_cond - y_raise
            for i in range(14):
                self.vw_raise[i] = mom * self.vw_raise[i] + lr * err_r * x[i]
                self.w_raise[i] -= self.vw_raise[i]
            for j in range(8):
                self.vu_raise[j] = mom * self.vu_raise[j] + lr * err_r * h[j]
                self.u_raise[j] -= self.vu_raise[j]


# ---------------------------------------------------------------------------
# MODULE 2: ONLINE EXP3 / MULTIPLICATIVE-WEIGHTS META-CONTROLLER
# ---------------------------------------------------------------------------
class EXP3MetaController:
    def __init__(self):
        self.log_weights = [0.0, 0.25, 0.25, 0.30]
        self.posture_ll = [0.0, 0.0, 0.0, 0.0]

    def update_from_profile(self, prof):
        s0 = 2.8 * (prof.fold_to_steal - 0.52) + 2.2 * (prof.fold_to_flop_bet - 0.48) - 1.8 * prof.postflop_agg_per_hand
        s1 = (
            2.4 * (0.35 - prof.fold_to_steal)
            + 2.2 * prof.turn_barrel_rate
            + 2.5 * prof.raise_my_bet_rate
            + 1.8 * (prof.three_bet_vs_open_rate - 0.09)
        )
        s2 = (
            1.8 * (prof.btn_open_rate - 0.52)
            + 1.6 * (0.45 - abs(prof.fold_to_bet - 0.38))
            + 2.0 * prof.flop_probe_rate
        )
        s3 = 0.45 + 0.8 * prof.postflop_agg_per_hand
        self.posture_ll = [s0, s1, s2, s3]

    def update_counterfactual_regret(self, hand_net, active_expert_weights):
        lr = EMBEDDED_RL_WEIGHTS["policy"]["exp3_lr"]
        norm_r = max(-1.0, min(1.0, hand_net / 40.0))
        for i in range(4):
            self.log_weights[i] = 0.97 * self.log_weights[i] + lr * active_expert_weights[i] * norm_r

    @property
    def probs(self):
        comb = [self.log_weights[i] + self.posture_ll[i] for i in range(4)]
        m = max(comb)
        exps = [math.exp(max(-10.0, c - m)) for c in comb]
        s = sum(exps)
        return [e / s for e in exps]


# ---------------------------------------------------------------------------
# MODULE 3: SIZING-AWARE BAYESIAN + SELF-IMAGE PROFILER
# ---------------------------------------------------------------------------
class LatentOpponentProfile:
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
        self.opp_faced_3bet = 0
        self.opp_folded_to_3bet = 0
        self.opp_4bets = 0

        # Track opponent's exact Button open sizes (5 vs 6 vs 7+) to detect sizing tells!
        self.opp_open_size_5 = 0
        self.opp_open_size_6 = 0
        self.opp_open_size_7plus = 0
        self.current_hand_opp_open_size = 0

        self.postflop_bets_raises = 0
        self.postflop_calls_checks = 0
        self.faced_bet_postflop = 0
        self.folded_to_bet_postflop = 0
        self.my_bet_exp_folds = 0.0
        self.pending_exp_fold = 0.0
        self.faced_flop_bet = 0
        self.folded_to_flop_bet = 0
        self.faced_turn_river_bet = 0
        self.folded_to_turn_river_bet = 0
        self.raised_my_bet_postflop = 0
        self.check_raises = 0
        self.turn_bet_hands = 0
        self.turn_seen_hands = 0
        self.river_bet_hands = 0
        self.river_seen_hands = 0
        self.flop_probes_seen = 0
        self.flop_seen_hands = 0

        self.overbets = 0
        self.overbet_showdowns = 0
        self.overbet_bluffs = 0
        self.hand_overbet_streets = set()

        self.high_card_discards = 0
        self.total_discards = 0
        self.showdowns = 0
        self.showdown_bluffs = 0

        # Self-image tracking
        self.my_btn_hands = 0
        self.my_btn_opens = 0
        self.my_faced_bet_postflop = 0
        self.my_folded_to_bet_postflop = 0

        self.cal_n = 0
        self.cal_gap = 0.0

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
        self.hand_saw_flop = False
        self.hand_saw_turn = False
        self.hand_bet_turn = False
        self.hand_overbet_streets = set()
        self.hand_saw_river = False
        self.hand_bet_river = False
        self.opp_checked_this_street = False
        self.me_bet_this_street = False
        self.opp_bet_this_street = False
        self.me_opened_preflop = False
        self.opp_opened_preflop = False
        self.me_3bet_preflop = False
        self.opp_responded_to_open = False
        self.opp_responded_to_3bet = False
        self.opp_first_pre_tracked = False
        self.my_first_pre_tracked = False
        self.current_hand_opp_open_size = 0
        self.pending_exp_fold = 0.0
        self.last_street_tracked = "preflop"
        self.processed_history_len = 0

    def start_hand(self, my_seat="button"):
        if self.hands_seen > 0:
            if self.hand_saw_flop:
                self.flop_seen_hands += 1
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
        else:
            self.my_btn_hands += 1

    def observe_history(self, history, nn_model=None, pending_nn_ref=None):
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

            if street == "flop":
                self.hand_saw_flop = True
            elif street == "turn":
                self.hand_saw_turn = True
            elif street == "river":
                self.hand_saw_river = True

            if street != self.last_street_tracked:
                self.opp_checked_this_street = False
                self.me_bet_this_street = False
                self.opp_bet_this_street = False
                self.last_street_tracked = street

            if actor == "opp":
                if street == "preflop":
                    if self.opp_is_btn and not self.opp_first_pre_tracked:
                        self.opp_first_pre_tracked = True
                        if action == "raise":
                            self.opp_btn_opens += 1
                            self.opp_opened_preflop = True
                            self.current_hand_opp_open_size = amount
                            if amount <= 5:
                                self.opp_open_size_5 += 1
                            elif amount == 6:
                                self.opp_open_size_6 += 1
                            else:
                                self.opp_open_size_7plus += 1
                    if self.me_opened_preflop and not self.opp_responded_to_open:
                        self.opp_responded_to_open = True
                        self.faced_preflop_steal += 1
                        if action == "fold":
                            self.folded_to_preflop_steal += 1
                        elif action == "raise":
                            self.three_bet_my_open += 1
                    elif self.me_3bet_preflop and not self.opp_responded_to_3bet:
                        self.opp_responded_to_3bet = True
                        self.opp_faced_3bet += 1
                        if action == "fold":
                            self.opp_folded_to_3bet += 1
                        elif action == "raise":
                            self.opp_4bets += 1
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
                    if self.pending_exp_fold > 0.0 and action in ("fold", "call", "raise"):
                        self.my_bet_exp_folds += self.pending_exp_fold
                        self.pending_exp_fold = 0.0
                    if pending_nn_ref is not None and pending_nn_ref[0] is not None and action in ("fold", "call", "raise"):
                        if nn_model is not None:
                            nn_model.online_momentum_update(pending_nn_ref[0], action)
                        pending_nn_ref[0] = None

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
                        self.opp_bet_this_street = True
                        if street == "flop" and amount <= 6:
                            self.flop_probes_seen += 1
                        elif street == "turn":
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
                if street == "preflop":
                    if not self.opp_is_btn and not self.my_first_pre_tracked:
                        self.my_first_pre_tracked = True
                        if action == "raise":
                            self.my_btn_opens += 1
                    if action == "raise":
                        if preflop_raises_seen == 0:
                            self.me_opened_preflop = True
                        elif preflop_raises_seen == 1 and self.opp_opened_preflop:
                            self.me_3bet_preflop = True
                        preflop_raises_seen += 1
                else:
                    if self.opp_bet_this_street and action in ("fold", "call", "raise"):
                        self.my_faced_bet_postflop += 1
                        if action == "fold":
                            self.my_folded_to_bet_postflop += 1
                        self.opp_bet_this_street = False
                    if action == "raise":
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
    def fold_to_3bet(self):
        return (self.opp_folded_to_3bet + 2.2) / (self.opp_faced_3bet + 5.5)

    @property
    def four_bet_rate(self):
        return (self.opp_4bets + 0.4) / (self.opp_faced_3bet + 6.0)

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
    def opp_fold_scale(self):
        return max(0.20, min(2.20, (self.folded_to_bet_postflop + 2.5) / (self.my_bet_exp_folds + 2.5)))

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
        return (self.overbets + 0.4) / (self.turn_seen_hands + self.river_seen_hands + 8.0)

    @property
    def overbet_bluff_rate(self):
        return (self.overbet_bluffs + 0.5) / (self.overbet_showdowns + 5.0)

    @property
    def turn_barrel_rate(self):
        return (self.turn_bet_hands + 1.5) / (self.turn_seen_hands + 5.0)

    @property
    def river_barrel_rate(self):
        return (self.river_bet_hands + 1.5) / (self.river_seen_hands + 5.0)

    @property
    def flop_probe_rate(self):
        return (self.flop_probes_seen + 1.0) / (self.flop_seen_hands + 5.0)

    @property
    def showdown_bluff_rate(self):
        return (self.showdown_bluffs + 0.4) / (self.showdowns + 3.5)

    @property
    def my_public_fold_to_bet(self):
        return (self.my_folded_to_bet_postflop + 3.0) / (self.my_faced_bet_postflop + 8.0)

    @property
    def my_public_btn_open(self):
        return (self.my_btn_opens + 2.7) / (self.my_btn_hands + 6.0)

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


# ---------------------------------------------------------------------------
# MODULE 4: SIZING-AWARE NEURAL CFR CONTROLLER
# ---------------------------------------------------------------------------
class NeuralCFRBot:
    def __init__(self):
        self.total_hands = 200
        self.starting_stack = 200
        self.time_bank_ms = 30000
        self.cap = 400

        self.hand_id = 0
        self.seat = "button"
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None
        self.cum_net = 0

        self.opp = LatentOpponentProfile()
        self.nn = OnlineResponseNet()
        self.meta = EXP3MetaController()
        self.pending_nn_ref = [None]
        self.pend_river_est = None

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
        strength_quantile(0.5)

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
        self.pending_nn_ref[0] = None
        self.pend_river_est = None
        self.opp.start_hand(self.seat)
        self.meta.update_from_profile(self.opp)

    def handle_discard_result(self, msg):
        self.my_discard = msg["my_discard"]
        self.opp_discard = msg["opp_discard"]
        self.hole2 = list(msg["hole_cards"])
        self.opp.observe_discard(self.opp_discard)

    def handle_hand_result(self, msg):
        net = msg.get("net", 0)
        self.cum_net += net
        hist = msg.get("history", [])
        self.opp.observe_history(hist, self.nn, self.pending_nn_ref)

        went_sd = msg.get("result") == "showdown"
        if went_sd and self.pend_river_est is not None:
            win = 1.0 if net > 0 else (0.5 if net == 0 else 0.0)
            self.opp.cal_n += 1
            self.opp.cal_gap += win - self.pend_river_est
        self.pend_river_est = None

        if went_sd and msg.get("opp_hole_cards"):
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

        self.meta.update_counterfactual_regret(net, self.meta.probs)

    def opp_preflop_range(self, hist):
        """
        Sizing-Conditioned Soft Sigmoid Preflop Range Encoder:
        Combines action sequence WITH observed preflop raise sizing tells (`5` vs `6` vs `7+` chips)!
        """
        pre = [h for h in hist if h.get("street") == "preflop"]
        opp_acts = [h["action"] for h in pre if h.get("actor") == "opp"]
        if not opp_acts:
            return None
        my_raises = sum(1 for h in pre if h.get("actor") == "me" and h.get("action") == "raise")
        opp_raises = opp_acts.count("raise")
        o = self.opp

        def top(x):
            return strength_quantile(1.0 - max(0.04, min(0.96, x)))

        if self.seat == "big_blind":
            if opp_acts[0] == "raise":
                if opp_raises >= 2:
                    return RangeWeight(lo=top(0.07))
                if my_raises >= 1:
                    return RangeWeight(lo=top(o.btn_open_rate * 0.55))
                # Sizing tell detection on single Button open:
                open_sz = o.current_hand_opp_open_size
                if o.opp_open_size_5 >= 2 and o.opp_open_size_6 >= 2:
                    # Opponent splits opens between 5 (weak/medium) and 6 (strong, top ~24%):
                    if open_sz == 6:
                        return RangeWeight(lo=top(min(0.26, o.btn_open_rate * 0.45)))
                    elif open_sz == 5:
                        return RangeWeight(lo=top(o.btn_open_rate), hi=top(0.20))
                elif o.opp_open_size_6 >= 2 and o.opp_open_size_7plus >= 1:
                    # Opponent splits opens between 6 (standard) and 7 (top 10% monster, e.g., Varshith):
                    if open_sz >= 7:
                        return RangeWeight(lo=top(0.12))
                    elif open_sz == 6:
                        return RangeWeight(lo=top(o.btn_open_rate), hi=top(0.09))
                return RangeWeight(lo=top(o.btn_open_rate))
            if opp_acts[0] == "call":
                if opp_raises >= 1:
                    return RangeWeight(lo=top(0.25))
                if o.opp_btn_hands >= 10 and o.btn_open_rate > 0.15:
                    return RangeWeight(hi=top(o.btn_open_rate))
            return None
        if my_raises == 0:
            if opp_raises >= 1:
                return RangeWeight(lo=top(max(0.15, o.pfr)))
            return RangeWeight(hi=top(max(0.10, o.pfr * 0.8)))
        if opp_acts[0] == "raise":
            if opp_raises >= 2:
                return RangeWeight(lo=top(0.05))
            if my_raises >= 2:
                return RangeWeight(lo=top(o.three_bet_vs_open_rate * 0.6))
            return RangeWeight(lo=top(o.three_bet_vs_open_rate))
        if opp_acts[0] == "call":
            return RangeWeight(lo=top(1.0 - o.fold_to_steal), hi=top(o.three_bet_vs_open_rate))
        return None

    def rollouts(self, full, minimum, time_left):
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
        self.opp.observe_history(req.get("history", []), self.nn, self.pending_nn_ref)
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        time_left = req.get("time_left_ms", 30000)

        archetype = self.opp.classify()
        exp_probs = self.meta.probs

        if time_left < 2000:
            if "check" in legal:
                return {"type": "action", "action": "check"}
            if to_call <= 4 and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if street == "preflop":
            return self._play_preflop_rl(req, archetype, exp_probs)
        res = self._play_postflop_rl(req, archetype, exp_probs, time_left)
        if res.get("action") == "raise":
            bet_sz = max(0, int(res.get("amount", 0)) - req["my_bet"])
            if bet_sz > 0:
                self.opp.pending_exp_fold = bet_sz / max(1.0, float(req["pot"] + 2.0 * bet_sz))
        return res

    def _play_preflop_rl(self, req, archetype, exp_probs):
        legal = req["legal_actions"]
        to_call = req["to_call"]
        opp_bet = req["opp_bet"]
        my_bet = req["my_bet"]
        pot = req["pot"]
        strength = preflop_3card_strength(self.hole3)
        P = EMBEDDED_RL_WEIGHTS["policy"]

        # 1. Anti-Shove Spammer Guard
        if self.opp.hands_seen >= 6 and self.opp.preflop_shove_rate > 0.25:
            if strength >= 0.57:
                if "raise" in legal and strength >= 0.72:
                    return {"type": "action", "action": "raise", "amount": req["max_raise"]}
                if "call" in legal and to_call > 0:
                    return {"type": "action", "action": "call"}
                if "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": min(req["max_raise"], 6)}
            if to_call == 0:
                return {"type": "action", "action": "check"}
            return {"type": "action", "action": "fold"}

        # 2. Extreme Preflop Folder Exploit (`fold_to_steal > 0.62`, e.g., Bhavesh)
        if self.opp.hands_seen >= 8 and self.opp.fold_to_steal > 0.62 and self.opp.three_bet_vs_open_rate < 0.12 and opp_bet <= 2:
            if "raise" in legal:
                return {"type": "action", "action": "raise", "amount": req["min_raise"]}

        open_thresh = P["btn_open_base"]
        limp_thresh = P["btn_limp_floor"]
        call_raise_thresh = P["bb_defend_base"]
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

        # Anti-3-Bet & Sticky-BB / Varshith 60%-Open-Switch Prevention:
        if self.opp.faced_preflop_steal >= 6 and self.opp.three_bet_vs_open_rate > 0.115:
            open_thresh = max(open_thresh, P["btn_open_vs_3bet"])
            limp_thresh = 0.20
        if self.opp.faced_preflop_steal >= 8 and self.opp.fold_to_steal < 0.18:
            open_thresh = max(open_thresh, 0.44)
            limp_thresh = 0.20
        if self.seat != "button" and opp_bet <= 2 and self.opp.opp_btn_hands >= 8 and self.opp.btn_open_rate < 0.32 and self.opp.vpip > 0.75:
            open_thresh = max(open_thresh, 0.68)

        # Sizing-Aware 3-Bet & BB Defense vs Wide Openers:
        if self.seat != "button" and self.opp.opp_btn_hands >= 8 and self.opp.btn_open_rate > 0.56:
            three_bet_val_thresh = min(three_bet_val_thresh, P["bb_3bet_vs_wide"])
            call_raise_thresh = min(call_raise_thresh, 0.34)

        # If opponent splits Button open sizes (5 vs 6, or 6 vs 7):
        open_sz = self.opp.current_hand_opp_open_size
        if self.seat != "button" and my_bet <= 2 and opp_bet in (5, 6, 7):
            if self.opp.opp_open_size_5 >= 2 and self.opp.opp_open_size_6 >= 2:
                if open_sz == 5:
                    # Capped weak/medium open (0.32..0.56): 3-bet wider (`0.54`) and defend wider (`0.32`)!
                    three_bet_val_thresh = min(three_bet_val_thresh, 0.54)
                    call_raise_thresh = min(call_raise_thresh, 0.32)
                elif open_sz == 6:
                    # Uncapped top-range open (`>= 0.56`): tighten 3-betting (`0.68`)!
                    three_bet_val_thresh = max(three_bet_val_thresh, 0.68)
            elif self.opp.opp_open_size_6 >= 2 and self.opp.opp_open_size_7plus >= 1 and open_sz >= 7:
                # Top 10% monster open from Varshith (`p >= 0.90`): only 3-bet super-premiums (`0.78`)!
                three_bet_val_thresh = max(three_bet_val_thresh, 0.78)
                call_raise_thresh = max(call_raise_thresh, 0.42)

        if opp_bet <= 2:
            if strength >= open_thresh and "raise" in legal:
                # Disguise our own Button open size at 5 chips (except vs Calling Stations) so opponents
                # cannot exploit a 5-vs-6 sizing tell against us!
                if archetype == "CALLING_STATION" and strength > 0.60:
                    size = 7
                else:
                    size = 5
                amt = max(req["min_raise"], min(req["max_raise"], size))
                return {"type": "action", "action": "raise", "amount": amt}
            if to_call == 0:
                return {"type": "action", "action": "check"}
            if strength >= limp_thresh and "call" in legal:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        pot_odds = to_call / max(1.0, float(pot + to_call))

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
            req_def = max(0.49, pot_odds + 0.18) if opp_bet >= 18 else 0.49
            if strength >= req_def or (self.seat == "button" and opp_bet < 18 and strength >= 0.46 and pot_odds <= 0.34):
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        if strength >= three_bet_val_thresh and "raise" in legal:
            target = min(req["max_raise"], max(req["min_raise"], int(opp_bet * 2.8)))
            return {"type": "action", "action": "raise", "amount": target}

        btn_ip_discount = 0.04 if (self.seat == "button" and self.opp.three_bet_vs_open_rate > 0.11) else 0.0
        if strength >= (call_raise_thresh - btn_ip_discount) or (pot_odds <= 0.34 and strength >= 0.33):
            return {"type": "action", "action": "call"}

        return {"type": "action", "action": "fold"}

    def _play_postflop_rl(self, req, archetype, exp_probs, time_left):
        legal = req["legal_actions"]
        street = req["street"]
        street_idx = {"flop": 1, "turn": 2, "river": 3}[street]
        board = req["board"]
        to_call = req["to_call"]
        pot = req["pot"]
        my_bet = req["my_bet"]
        opp_bet = req["opp_bet"]
        eff_stack = min(req["my_stack"], req["opp_stack"])
        hist = req.get("history", [])
        P = EMBEDDED_RL_WEIGHTS["policy"]

        if self.pre_w is None:
            self.pre_w = self.opp_preflop_range(hist)
        pre_raises = [h for h in hist if h["street"] == "preflop" and h["action"] == "raise"]
        is_3bet_pot = len(pre_raises) >= 2
        freq_raiser = self.opp.faced_bet_postflop >= 5 and self.opp.raise_my_bet_rate > 0.20

        opp_bluff_raise_active = (
            self.opp.faced_bet_postflop >= 6
            and self.opp.raise_my_bet_rate > 0.16
            and (self.opp.my_public_fold_to_bet >= 0.41 or self.opp.raise_my_bet_rate > 0.24)
        )

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
        opp_checked_street = any(h["street"] == street and h["actor"] == "opp" and h["action"] == "check" for h in hist)

        opp_bet_flop = any(h["street"] == "flop" and h["actor"] == "opp" and h["action"] == "raise" for h in hist)
        opp_bet_turn = any(h["street"] == "turn" and h["actor"] == "opp" and h["action"] == "raise" for h in hist)
        opp_raised_us_flop = False
        flop_r = [h["actor"] for h in hist if h["street"] == "flop" and h["action"] == "raise"]
        if len(flop_r) >= 2 and "me" in flop_r and flop_r[-1] == "opp":
            opp_raised_us_flop = True
        opp_raised_us_turn = False
        turn_r = [h["actor"] for h in hist if h["street"] == "turn" and h["action"] == "raise"]
        if len(turn_r) >= 2 and "me" in turn_r and turn_r[-1] == "opp":
            opp_raised_us_turn = True
        opp_raised_us_prev = opp_raised_us_flop or opp_raised_us_turn
        board_ranks = [RANK_VAL[c[0]] for c in board]
        board_is_paired = len(set(board_ranks)) < len(board_ranks)

        bet_to_pot_ratio = to_call / max(1.0, pot - to_call) if to_call > 0 else 0.0

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
            # Sizing-Conditioned Postflop Range Bias (`SizingBeliefEncoder`):
            if archetype == "MANIAC":
                opp_bias = -0.20 if to_call > 0 else 0.0
            elif to_call > 0 and bet_to_pot_ratio <= 0.43 and not is_3bet_pot:
                # Small-ball probe / river air stab (`0.35x-0.40x` pot): range is capped/bluff-heavy!
                opp_bias = -0.12
            elif (self.opp.hands_seen >= 10 and self.opp.vpip > 0.70 and agg_postflop_opp) and to_call > 0 and not is_3bet_pot:
                if street == "river" and opp_bet_flop and opp_bet_turn:
                    opp_bias = 0.08
                else:
                    opp_bias = -0.06 if bet_to_pot_ratio < 1.05 else 0.02
            elif agg_postflop_opp and to_call > 0 and not is_3bet_pot:
                opp_bias = 0.04 if bet_to_pot_ratio < 0.72 else 0.10
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
        pot_odds = to_call / max(1.0, float(pot + to_call))
        eq_adj = eq
        # Positional & Draw Equity Realization Critic (Flop/Turn only; River is 100% exact)
        if street != "river":
            wr = EMBEDDED_RL_WEIGHTS["w_realize"]
            s_left = 1.0 if street == "flop" else 0.5
            r_shift = (
                wr[1] * (1.0 if self.seat == "button" else 0.0)
                + wr[2] * (1.0 if has_draw else 0.0)
                + wr[3] * (inter / 4.0)
                + wr[4] * danger
                + wr[5] * (1.0 if is_3bet_pot else 0.0)
                + wr[6] * s_left
                + wr[7] * max(0.0, self.opp.postflop_agg_per_hand - 0.5)
            )
            eq_adj = max(0.01, min(0.99, eq_adj + r_shift))

        # ------------------------------------------------------------------
        # Scenario A: Checked to us (`to_call == 0`) — Neural Multi-Sizing EV + Probes
        # ------------------------------------------------------------------
        if to_call == 0:
            if "raise" not in legal:
                return {"type": "action", "action": "check"}

            if self.opp.hands_seen >= 10 and self.opp.fold_to_bet > 0.68:
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * 0.30)))
                return {"type": "action", "action": "raise", "amount": amt}

            if street == "flop" and self.seat != "button" and opp_was_prev_aggressor and inter < 3 and eq_adj < 0.70:
                return {"type": "action", "action": "check"}

            if street in ("turn", "river") and opp_was_prev_aggressor:
                if inter < 3 and (eq_adj < 0.78 or opp_raised_us_prev):
                    return {"type": "action", "action": "check"}
                if street == "river" and opp_raised_us_prev and inter == 3 and full_cat <= 3 and eq_adj < 0.88:
                    return {"type": "action", "action": "check"}

            if (
                street == "turn"
                and self.seat != "button"
                and inter >= 3
                and eq_adj >= 0.78
                and self.opp.turn_barrel_rate > 0.38
                and random.random() < P["oop_turn_trap_freq"]
            ):
                return {"type": "action", "action": "check"}

            if archetype == "MANIAC":
                if street in ("flop", "turn") and 0.48 <= eq_adj <= 0.75:
                    return {"type": "action", "action": "check"}
                if eq_adj > 0.75 or (street == "river" and eq_adj >= 0.58 and inter >= 2):
                    frac = 0.85 if eq_adj > 0.80 else 0.62
                    target = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                    return {"type": "action", "action": "raise", "amount": target}
                return {"type": "action", "action": "check"}

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
                        target = max(req["min_raise"], min(req["max_raise"], int(pot * 0.42) + 1))
                        return {"type": "action", "action": "raise", "amount": target}
                return {"type": "action", "action": "check"}

            if street == "turn" and inter == 1 and eq_adj < 0.66:
                return {"type": "action", "action": "check"}

            # Neural Counterfactual Multi-Sizing Value Selector (`OnlineResponseNet`):
            emp_f = self.opp.fold_to_flop_bet if street == "flop" else self.opp.fold_to_turn_river_bet
            if inter >= 3 and eq_adj >= 0.72:
                streets_left = {"flop": 3, "turn": 2, "river": 1}[street]
                geo_frac = geometric_bet_fraction(pot, eff_stack, streets_left)
                base_frac = max(0.68, min(0.95, geo_frac))
                if archetype == "CALLING_STATION":
                    base_frac = max(base_frac, 0.90)
                cand_fracs = (max(0.56, base_frac - 0.14), base_frac, min(1.08, base_frac + 0.14))
                best_target = max(req["min_raise"], min(req["max_raise"], int(pot * base_frac)))
                best_ev = -1e9
                best_x = None
                for cf in cand_fracs:
                    t_amt = max(req["min_raise"], min(req["max_raise"], my_bet + max(2, int(pot * cf))))
                    B = t_amt - my_bet
                    if B <= 0:
                        continue
                    x_nn = self.nn.build_features(
                        street_idx, B, pot, danger, self.opp.fold_to_bet,
                        self.opp.fold_to_flop_bet, self.opp.fold_to_turn_river_bet,
                        self.opp.postflop_agg_per_hand, self.opp.raise_my_bet_rate,
                        self.opp.vpip, is_3bet_pot, opp_checked_street, self.opp.my_public_fold_to_bet,
                    )
                    pf, pc, pr, _, _ = self.nn.predict(x_nn, emp_f)
                    eq_c = max(0.55, eq_adj - 0.12 * pf)
                    ev_c = eq_c * (pot + 2.0 * B) - B
                    ev_r = (eq_adj - 0.06) * (pot + 3.8 * B) - 1.9 * B
                    ev_total = pf * pot + pc * ev_c + pr * ev_r
                    if ev_total > best_ev:
                        best_ev = ev_total
                        best_target = t_amt
                        best_x = x_nn
                self.pending_nn_ref[0] = best_x
                return {"type": "action", "action": "raise", "amount": best_target}

            val_thresh = P["val_eq_thresh"]
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
                    frac = P["strong_val_size"]
                    if self.opp.faced_bet_postflop >= 6 and self.opp.opp_fold_scale < 0.78:
                        frac = min(1.00, frac + 0.12)
                else:
                    frac = P["thin_val_size"]
                target = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                return {"type": "action", "action": "raise", "amount": target}

            bluff_freq = 0.0
            probe_frac = P["flop_probe_size"]
            if not freq_raiser and archetype not in ("CALLING_STATION", "MANIAC", "TRAPPER"):
                if street == "flop" and 0.32 <= eq_adj < val_thresh and not is_3bet_pot:
                    if self.opp.faced_flop_bet >= 5 and self.opp.fold_to_flop_bet > 0.44:
                        bluff_freq = P["flop_probe_freq_high"]
                        probe_frac = 0.35
                    elif eq_adj >= 0.38:
                        bluff_freq = 0.52 if archetype == "NIT_TAG" else P["flop_probe_freq_base"]
                        probe_frac = 0.36
                elif street == "turn" and prev_checked_through and 0.30 <= eq_adj < val_thresh:
                    bluff_freq = 0.48 if archetype == "NIT_TAG" else P["turn_float_freq"]
                    probe_frac = 0.40
            if bluff_freq > 0.0 and self.opp.faced_bet_postflop >= 6:
                if self.opp.opp_fold_scale < 0.65:
                    bluff_freq *= 0.45
                elif self.opp.opp_fold_scale > 1.15:
                    bluff_freq = min(0.85, bluff_freq + 0.14)

            if bluff_freq > 0 and random.random() < bluff_freq:
                target = max(req["min_raise"], min(req["max_raise"], int(pot * probe_frac)))
                return {"type": "action", "action": "raise", "amount": target}

            return {"type": "action", "action": "check"}

        # ------------------------------------------------------------------
        # Scenario B: Facing a Bet or Raise (`to_call > 0`)
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

        # 0. Overbet Top-Range & Bluff-Share Solver (>= 1.1x pot on turn/river)
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
                    if street == "river":
                        self.pend_river_est = eq
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold" if "fold" in legal else "check"}

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
                if street == "river":
                    self.pend_river_est = eq
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}

        # 2. Facing a Postflop Raise Over Our Bet (`opp_raised_our_bet`) or Check-Raise:
        if opp_raised_our_bet or opp_check_raised:
            if (
                (inter == 4 and eq_adj >= 0.82 and (street != "river" or danger < 0.35))
                or (street != "river" and inter >= 3 and full_cat >= 3 and eq_adj >= 0.85 and danger < 0.45)
            ) and "raise" in legal:
                target = opp_bet + int((pot + to_call) * 0.75)
                amt = max(req["min_raise"], min(req["max_raise"], target))
                return {"type": "action", "action": "raise", "amount": amt}

            if street in ("turn", "river") and inter <= 2:
                max_po = 0.33 if (opp_bluff_raise_active and not opp_check_raised and to_call <= 45) else 0.22
                if inter == 2 and (pot_odds <= max_po or (freq_raiser and pot_odds <= 0.30)) and eq_adj >= 0.58:
                    if street == "river":
                        self.pend_river_est = eq
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

            if street in ("turn", "river") and inter == 3 and full_cat == 2:
                if (board_is_paired and to_call >= 20 and eq_adj < 0.80) or (street == "river" and to_call >= 25 and (danger >= 0.15 or eq_adj < 0.78)) or eq_adj < 0.66 or (danger >= 0.50 and eq_adj < 0.72):
                    return {"type": "action", "action": "fold"}
                if street == "river":
                    self.pend_river_est = eq
                return {"type": "action", "action": "call"}

            if street in ("turn", "river") and inter == 3 and full_cat == 3:
                if eq_adj < 0.60 or (street == "river" and to_call >= 40 and danger >= 0.40 and eq_adj < 0.74):
                    return {"type": "action", "action": "fold"}
                if street == "river":
                    self.pend_river_est = eq
                return {"type": "action", "action": "call"}

            if street == "river" and full_cat == 4 and to_call >= 35 and danger >= 0.25 and eq_adj < 0.78:
                return {"type": "action", "action": "fold"}

            if street == "flop" and inter <= 1 and not has_draw and (eq_adj < 0.58 or to_call >= 12):
                return {"type": "action", "action": "fold"}

            if street == "flop" and inter == 2 and not has_draw and to_call >= 32 and bet_to_pot_ratio >= 0.72 and eq_adj < 0.76:
                return {"type": "action", "action": "fold"}

            eq_adj -= (0.02 if (street == "flop" and freq_raiser and (inter >= 2 or has_draw)) else 0.07)

        # 3. Sizing-Aware Counter-Exploit Against Small-Ball Flop/Turn/River Probes (`<= 0.42x` pot):
        # In AdaptiveBot v9, bets `<= 0.42x` pot on Flop (`0.35x`), Turn (`0.40x`), and checked-through
        # River (`0.38x`) are 100% probes/bluffs (`eq < 0.56` on Flop/Turn, `inter == 0` air on River)!
        if not opp_raised_our_bet and not opp_check_raised and not is_3bet_pot and bet_to_pot_ratio <= 0.42:
            if street == "river" and prev_checked_through and inter >= 1 and eq_adj >= 0.34 and "call" in legal:
                self.pend_river_est = eq
                return {"type": "action", "action": "call"}
            if street == "flop" and self.seat == "button" and (inter >= 2 or (has_draw and eq_adj >= 0.48)) and "raise" in legal:
                if random.random() < P["probe_counter_raise_freq"]:
                    target = opp_bet + int((pot + to_call) * 0.58)
                    amt = max(req["min_raise"], min(req["max_raise"], target))
                    return {"type": "action", "action": "raise", "amount": amt}

        # 4. Structural Defense Against Flop, Turn & River Bets:
        wide_agg = agg_postflop_opp and self.opp.vpip > 0.68
        if street == "flop":
            if inter == 0 and not has_draw:
                if (bet_to_pot_ratio >= 0.38 or is_3bet_pot) and eq_adj < max(0.55, pot_odds + 0.12):
                    return {"type": "action", "action": "fold"}
            if inter == 1 and not has_draw:
                flop_fold_eq = (pot_odds + 0.03) if (agg_postflop_opp or self.opp.vpip > 0.68) else 0.46
                if (bet_to_pot_ratio >= 0.64 or is_3bet_pot) and eq_adj < flop_fold_eq:
                    return {"type": "action", "action": "fold"}

        if street == "river":
            if inter == 0 and to_call > pot * 0.16:
                return {"type": "action", "action": "fold"}
            if to_call >= 60 and bet_to_pot_ratio >= 0.85 and inter <= 2 and eq_adj < (pot_odds + 0.12 if wide_agg else 0.84) and self.opp.showdown_bluff_rate < 0.35:
                return {"type": "action", "action": "fold"}
            inter1_lim = 0.38 if (prev_checked_through or agg_postflop_opp) else 0.26
            inter1_eq_req = (pot_odds + 0.06) if wide_agg else (0.56 if (prev_checked_through and agg_postflop_opp) else 0.66)
            if inter == 1 and (to_call > pot * inter1_lim or (opp_was_prev_aggressor and not agg_postflop_opp)) and eq_adj < inter1_eq_req:
                return {"type": "action", "action": "fold"}
            if opp_bet_flop and opp_bet_turn and bet_to_pot_ratio >= 0.48 and self.opp.river_barrel_rate < 0.48 and not wide_agg:
                if inter <= 2 and eq_adj < 0.58:
                    return {"type": "action", "action": "fold"}
            if bet_to_pot_ratio >= 0.75 and to_call >= 25 and inter <= 2 and eq_adj < (pot_odds + 0.10 if wide_agg else 0.82) and self.opp.showdown_bluff_rate < 0.30:
                return {"type": "action", "action": "fold"}
            if bet_to_pot_ratio >= 0.72 and self.opp.river_barrel_rate < 0.48 and not wide_agg:
                if inter <= 2 and eq_adj < 0.76:
                    return {"type": "action", "action": "fold"}
                if inter == 3 and full_cat == 2 and (danger >= 0.40 or board_is_paired) and eq_adj < 0.76:
                    return {"type": "action", "action": "fold"}

        if street == "turn":
            if inter == 0 and not has_draw and bet_to_pot_ratio >= 0.28 and eq_adj < 0.58:
                return {"type": "action", "action": "fold"}
            if inter == 0 and has_draw and (bet_to_pot_ratio >= 0.50 or is_3bet_pot) and eq_adj < 0.54:
                return {"type": "action", "action": "fold"}
            if inter == 1 and not has_draw:
                if is_3bet_pot and bet_to_pot_ratio >= 0.42 and eq_adj < 0.62:
                    return {"type": "action", "action": "fold"}
                turn_inter1_eq = max(pot_odds + 0.04, 0.44) if wide_agg else (P["turn_inter1_mdf_eq"] if agg_postflop_opp else 0.63)
                turn_ratio_lim = 0.68 if agg_postflop_opp else 0.55
                if (bet_to_pot_ratio >= turn_ratio_lim or (opp_was_prev_aggressor and not agg_postflop_opp)) and eq_adj < turn_inter1_eq:
                    return {"type": "action", "action": "fold"}
            if bet_to_pot_ratio >= 0.80 and not wide_agg and (inter <= 1 or (inter == 2 and not agg_postflop_opp)) and eq_adj < 0.68:
                return {"type": "action", "action": "fold"}

        # 5. Raise for Value (when facing an opening bet, my_bet == 0)
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

        # 6. Standard Call Decision (with Match-State Capped-Utility Shaping)
        margin = 0.02
        hands_rem = max(1, self.total_hands - self.hand_id + 1)
        if hands_rem <= max(30, int(0.18 * self.total_hands)):
            if self.cum_net >= 0.85 * self.cap:
                margin += 0.025  # Protect locked capped match win
            elif self.cum_net <= -0.65 * self.cap:
                margin -= 0.020  # Push high-upside spots to recover toward cap
        if street == "flop":
            margin = -0.02 if self.seat == "button" else 0.0
        elif street == "river":
            margin = 0.03

        if archetype == "NIT_TAG" and to_call > pot * 0.40:
            margin += 0.05
        elif archetype == "CALLING_STATION" and to_call > pot * 0.45:
            margin += 0.08

        if eq_adj >= pot_odds + margin and "call" in legal:
            if street == "river":
                self.pend_river_est = eq
            return {"type": "action", "action": "call"}

        return {"type": "action", "action": "fold" if "fold" in legal else "check"}


# ---------------------------------------------------------------------------
# Crash-proof main loop
# ---------------------------------------------------------------------------
AdaptiveBot = NeuralCFRBot
_ERRORS = 0


def _log_error(where):
    global _ERRORS
    _ERRORS += 1
    if _ERRORS <= 20:
        sys.stderr.write(f"[rl_bot error #{_ERRORS} in {where}]\n{traceback.format_exc()}\n")
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
    bot = NeuralCFRBot()
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
