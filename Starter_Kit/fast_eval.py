"""
Ultra-fast Hand Evaluator, Precomputed 1,755 Canonical 3-Card Preflop Evaluator,
Post-Flop Discard Optimizer, Exact Zero-Variance Combinatorial River Solver,
Pre-Evaluated Turn Rollout Engine, and Board Texture Danger Analyzer (v6.2 Championship Engine).
Uses only Python 3.11 standard library.
"""

import random
from collections import Counter
from itertools import combinations
from preflop_table import PREFLOP_TABLE_1755

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_VAL = {s: i for i, s in enumerate(SUITS)}
ALL_CARDS = tuple(r + s for r in RANKS for s in SUITS)
CARD_IDX = {c: i for i, c in enumerate(ALL_CARDS)}

CARD_RANK = tuple(RANK_VAL[c[0]] for c in ALL_CARDS)
CARD_SUIT = tuple(SUIT_VAL[c[1]] for c in ALL_CARDS)

COMBOS_7_5 = tuple(combinations(range(7), 5))
COMBOS_6_5 = tuple(combinations(range(6), 5))


def eval5_idx(c0, c1, c2, c3, c4):
    """
    Fast 5-card evaluator taking 5 integer card indices (0..51).
    Returns a single integer score (higher is strictly better).
    """
    r0, r1, r2, r3, r4 = CARD_RANK[c0], CARD_RANK[c1], CARD_RANK[c2], CARD_RANK[c3], CARD_RANK[c4]
    s_flush = (CARD_SUIT[c0] == CARD_SUIT[c1] == CARD_SUIT[c2] == CARD_SUIT[c3] == CARD_SUIT[c4])

    # Sort 5 ranks descending using fast sorting network
    if r0 < r1: r0, r1 = r1, r0
    if r3 < r4: r3, r4 = r4, r3
    if r2 < r4: r2, r4 = r4, r2
    if r2 < r3: r2, r3 = r3, r2
    if r0 < r3: r0, r3 = r3, r0
    if r0 < r2: r0, r2 = r2, r0
    if r1 < r4: r1, r4 = r4, r1
    if r1 < r3: r1, r3 = r3, r1
    if r1 < r2: r1, r2 = r2, r1

    # Check if all 5 ranks are distinct
    if r0 != r1 and r1 != r2 and r2 != r3 and r3 != r4:
        straight_high = 0
        if r0 - r4 == 4:
            straight_high = r0
        elif r0 == 14 and r1 == 5:
            straight_high = 5

        if straight_high:
            if s_flush:
                return (8 << 20) | straight_high
            return (4 << 20) | straight_high
        if s_flush:
            return (5 << 20) | (r0 << 16) | (r1 << 12) | (r2 << 8) | (r3 << 4) | r4
        return (r0 << 16) | (r1 << 12) | (r2 << 8) | (r3 << 4) | r4

    # Four of a kind
    if r0 == r3:
        return (7 << 20) | (r0 << 4) | r4
    if r1 == r4:
        return (7 << 20) | (r1 << 4) | r0

    # Full house or Three of a kind
    if r0 == r2:
        if r3 == r4:
            return (6 << 20) | (r0 << 4) | r3
        return (3 << 20) | (r0 << 8) | (r3 << 4) | r4
    if r1 == r3:
        return (3 << 20) | (r1 << 8) | (r0 << 4) | r4
    if r2 == r4:
        if r0 == r1:
            return (6 << 20) | (r2 << 4) | r0
        return (3 << 20) | (r2 << 8) | (r0 << 4) | r1

    # Two pair or One pair
    if r0 == r1:
        if r2 == r3:
            return (2 << 20) | (r0 << 8) | (r2 << 4) | r4
        if r3 == r4:
            return (2 << 20) | (r0 << 8) | (r3 << 4) | r2
        return (1 << 20) | (r0 << 12) | (r2 << 8) | (r3 << 4) | r4
    if r1 == r2:
        if r3 == r4:
            return (2 << 20) | (r1 << 8) | (r3 << 4) | r0
        return (1 << 20) | (r1 << 12) | (r0 << 8) | (r3 << 4) | r4
    if r2 == r3:
        return (1 << 20) | (r2 << 12) | (r0 << 8) | (r1 << 4) | r4
    return (1 << 20) | (r3 << 12) | (r0 << 8) | (r1 << 4) | r2


def eval7_idx(cards7):
    """Best 5-card hand score out of 7 card indices."""
    best = 0
    for i0, i1, i2, i3, i4 in COMBOS_7_5:
        s = eval5_idx(cards7[i0], cards7[i1], cards7[i2], cards7[i3], cards7[i4])
        if s > best:
            best = s
    return best


def eval_cards(hole_strs, board_strs):
    """Evaluate best 5-card hand from string cards (hole + board, total 5, 6, or 7)."""
    idxs = [CARD_IDX[c] for c in hole_strs] + [CARD_IDX[c] for c in board_strs]
    n = len(idxs)
    if n == 7:
        return eval7_idx(idxs)
    elif n == 6:
        best = 0
        for i0, i1, i2, i3, i4 in COMBOS_6_5:
            s = eval5_idx(idxs[i0], idxs[i1], idxs[i2], idxs[i3], idxs[i4])
            if s > best:
                best = s
        return best
    elif n == 5:
        return eval5_idx(idxs[0], idxs[1], idxs[2], idxs[3], idxs[4])
    return 0


def canonical_3card_key(hole_strs):
    """Maps any 3 hole cards to their exact 1,755 canonical isomorphism key (r0, r1, r2, sp)."""
    idxs = sorted([CARD_IDX[c] for c in hole_strs], key=lambda i: CARD_RANK[i], reverse=True)
    r0, r1, r2 = CARD_RANK[idxs[0]], CARD_RANK[idxs[1]], CARD_RANK[idxs[2]]
    s0, s1, s2 = CARD_SUIT[idxs[0]], CARD_SUIT[idxs[1]], CARD_SUIT[idxs[2]]
    if r0 == r1 == r2:
        sp = 0
    elif r0 == r1:
        sp = 1 if (s2 == s0 or s2 == s1) else 0
    elif r1 == r2:
        sp = 1 if (s0 == s1 or s0 == s2) else 0
    else:
        if s0 == s1 == s2:
            sp = 4
        elif s0 == s1:
            sp = 3
        elif s0 == s2:
            sp = 2
        elif s1 == s2:
            sp = 1
        else:
            sp = 0
    return (r0, r1, r2, sp)


def preflop_3card_strength(hole_strs):
    """O(1) Lookup in the precomputed 1,755 Canonical 3-Card Equity & Synergy Table."""
    key = canonical_3card_key(hole_strs)
    return PREFLOP_TABLE_1755.get(key, 0.45)


def hole_interaction_level(hole_strs, board_strs):
    """
    Board-Relative Hand Interaction Classifier:
      0 = High card or playing the board / board-only pair(s)
      1 = Underpair, Bottom Pair, or Middle Pair
      2 = Top Pair (1 hole card pairs highest board rank) or Pocket Overpair (even on a paired board!)
      3 = True Hole Two Pair (both hole cards pair two distinct board ranks) OR Trips/Set (using hole card)
      4 = Straight, Flush, Full House (using hole cards), or Quads+
    """
    full_score = eval_cards(hole_strs, board_strs)
    cat = full_score >> 20
    if cat == 0:
        return 0

    board_ranks = sorted([RANK_VAL[c[0]] for c in board_strs], reverse=True)
    hole_ranks = sorted([RANK_VAL[c[0]] for c in hole_strs], reverse=True)
    b_counts = Counter(board_ranks)
    max_board_rank = board_ranks[0]

    if len(board_strs) == 5:
        board_only_score = eval_cards([], board_strs)
        if (full_score >> 8) == (board_only_score >> 8):
            return 0

    if cat in (4, 5, 7, 8):
        return 4

    if cat == 6:
        trips_rank = (full_score >> 4) & 0xF
        pair_rank = full_score & 0xF
        if trips_rank in hole_ranks:
            return 4
        if hole_ranks[0] == hole_ranks[1] and hole_ranks[0] == pair_rank:
            return 3 if pair_rank > max_board_rank else 2
        if pair_rank in hole_ranks and b_counts.get(pair_rank, 0) == 1:
            return 3 if pair_rank == max_board_rank else 2
        return 0

    if cat == 3:
        trips_rank = (full_score >> 8) & 0xF
        if trips_rank in hole_ranks:
            return 3
        return 0

    if cat == 2:
        matched_unpaired_board = sum(1 for hr in set(hole_ranks) if b_counts.get(hr, 0) == 1)
        if matched_unpaired_board >= 2:
            return 3
        if hole_ranks[0] == hole_ranks[1]:
            if hole_ranks[0] > max_board_rank:
                return 2
            return 1
        if matched_unpaired_board == 1:
            matched_r = max(hr for hr in hole_ranks if b_counts.get(hr, 0) == 1)
            if matched_r >= max_board_rank:
                return 2
            return 1
        return 0

    if b_counts.get((full_score >> 12) & 0xF, 0) >= 2:
        return 0
    pair_rank = (full_score >> 12) & 0xF
    if pair_rank >= max_board_rank:
        return 2
    return 1


def has_strong_draw(hole_strs, board_strs):
    """Returns True if hole_strs + board_strs has a 4-flush or 4-straight draw on Flop/Turn."""
    if len(board_strs) >= 5:
        return False
    all_c = list(hole_strs) + list(board_strs)
    suits = [c[1] for c in all_c]
    for s in "shdc":
        if suits.count(s) >= 4 and any(c[1] == s for c in hole_strs):
            return True
    ranks = sorted(set(RANK_VAL[c[0]] for c in all_c))
    if 14 in ranks:
        ranks = [1] + ranks
    hole_r = {RANK_VAL[c[0]] for c in hole_strs}
    for i in range(len(ranks)):
        window = [r for r in ranks if 0 <= r - ranks[i] <= 4]
        if len(window) >= 4 and any(hr in window for hr in hole_r):
            return True
    return False


def board_danger_level(board_strs):
    """
    Returns a danger score in [0.0, 1.0] measuring how wet/dangerous the board is
    for 1-pair or bare 2-pair hands (3-flush, 4-flush, 4-straight, paired board).
    """
    if len(board_strs) < 3:
        return 0.0
    suits = [c[1] for c in board_strs]
    ranks = sorted(set(RANK_VAL[c[0]] for c in board_strs))
    if 14 in ranks:
        ranks = [1] + ranks

    danger = 0.0
    max_suit = max(suits.count(s) for s in "shdc")
    if max_suit >= 4:
        danger += 0.55
    elif max_suit == 3:
        danger += 0.25

    max_conn = 1
    for i in range(len(ranks)):
        cnt = sum(1 for r in ranks if 0 <= r - ranks[i] <= 4)
        if cnt > max_conn:
            max_conn = cnt
    if max_conn >= 4:
        danger += 0.40
    elif max_conn == 3 and len(board_strs) >= 4:
        danger += 0.15

    if len(set(c[0] for c in board_strs)) < len(board_strs):
        danger += 0.15

    return min(1.0, danger)


def evaluate_discard_options(hole3_strs, flop3_strs, num_rollouts=120, opp_vpip=0.65):
    """
    Evaluates all 3 flop discard choices, combining:
      1. Expected Showdown Equity against opponent's VPIP-filtered range and optimal discard
      2. Nut-Potential ($P_{pot}$) Implied-Odds Bonus
      3. Deterministic Flop Made-Hand & Kicker Tie-Breaker
    """
    hole_idxs = [CARD_IDX[c] for c in hole3_strs]
    flop_idxs = [CARD_IDX[c] for c in flop3_strs]
    dead_set = set(hole_idxs + flop_idxs)
    rem_deck = [i for i in range(52) if i not in dead_set]

    options = [
        (hole3_strs[0], (hole_idxs[1], hole_idxs[2])),
        (hole3_strs[1], (hole_idxs[0], hole_idxs[2])),
        (hole3_strs[2], (hole_idxs[0], hole_idxs[1])),
    ]

    wins = [0.0, 0.0, 0.0]
    nut_hits = [0.0, 0.0, 0.0]
    total_w = 0.0
    b0, b1, b2 = flop_idxs[0], flop_idxs[1], flop_idxs[2]

    for _ in range(num_rollouts):
        sample = random.sample(rem_deck, 5)
        opp0, opp1, opp2, turn, river = sample[0], sample[1], sample[2], sample[3], sample[4]

        opp_r_max = max(CARD_RANK[opp0], CARD_RANK[opp1], CARD_RANK[opp2])
        w = 1.0
        if opp_vpip < 0.50 and opp_r_max < 10 and CARD_RANK[opp0] != CARD_RANK[opp1]:
            w = 0.35

        opp_pairs = ((opp1, opp2), (opp0, opp2), (opp0, opp1))
        best_opp_flop = -1
        opp_keep = opp_pairs[0]
        for ok0, ok1 in opp_pairs:
            f_score = eval5_idx(ok0, ok1, b0, b1, b2)
            s_ok0, s_ok1 = CARD_SUIT[ok0], CARD_SUIT[ok1]
            if s_ok0 == s_ok1:
                suit_cnt = 2 + (CARD_SUIT[b0] == s_ok0) + (CARD_SUIT[b1] == s_ok0) + (CARD_SUIT[b2] == s_ok0)
                if suit_cnt == 4:
                    f_score = max(f_score, (1 << 20) | (13 << 12))
            if f_score > best_opp_flop:
                best_opp_flop = f_score
                opp_keep = (ok0, ok1)

        opp_final = eval7_idx((opp_keep[0], opp_keep[1], b0, b1, b2, turn, river))
        total_w += w

        for idx in range(3):
            k0, k1 = options[idx][1]
            my_final = eval7_idx((k0, k1, b0, b1, b2, turn, river))
            if my_final > opp_final:
                wins[idx] += w
                if (my_final >> 20) >= 3:
                    nut_hits[idx] += w
            elif my_final == opp_final:
                wins[idx] += 0.5 * w

    equities = {options[i][0]: wins[i] / max(total_w, 1e-9) for i in range(3)}
    scores = []
    for i in range(3):
        k0, k1 = options[i][1]
        flop_now = eval5_idx(k0, k1, b0, b1, b2)
        flop_tiebreak = (flop_now >> 12) * 0.00015 + (CARD_RANK[k0] + CARD_RANK[k1]) * 0.0004
        scores.append((wins[i] + 0.035 * nut_hits[i]) / max(total_w, 1e-9) + flop_tiebreak)
    best_i = max(range(3), key=lambda i: scores[i])
    return options[best_i][0], equities[options[best_i][0]], equities


def exact_river_equity(hole_idxs, board_idxs, rem_deck, opp_disc_idx, opp_range_bias=0.0):
    """
    Zero-Variance Combinatorial Enumeration over opponent 2-card hands on the River!
    Precomputes single-card flop-discard scores in 43 calls and my_score in 1 call.
    """
    n_rem = len(rem_deck)
    b0, b1, b2, b3, b4 = board_idxs[0], board_idxs[1], board_idxs[2], board_idxs[3], board_idxs[4]
    my_score = eval7_idx((hole_idxs[0], hole_idxs[1], b0, b1, b2, b3, b4))

    opp_disc_rank = CARD_RANK[opp_disc_idx] if opp_disc_idx is not None else 2
    opp_disc_suit = CARD_SUIT[opp_disc_idx] if opp_disc_idx is not None else -1

    alt_flop_score = {}
    if opp_disc_idx is not None:
        for c in rem_deck:
            alt_flop_score[c] = eval5_idx(c, opp_disc_idx, b0, b1, b2)

    flop_suit_counts = [0, 0, 0, 0]
    flop_suit_counts[CARD_SUIT[b0]] += 1
    flop_suit_counts[CARD_SUIT[b1]] += 1
    flop_suit_counts[CARD_SUIT[b2]] += 1

    total_weight = 0.0
    weighted_wins = 0.0

    for i in range(n_rem):
        opp0 = rem_deck[i]
        r0 = CARD_RANK[opp0]
        s0 = CARD_SUIT[opp0]
        alt0 = alt_flop_score.get(opp0, 0)
        for j in range(i + 1, n_rem):
            opp1 = rem_deck[j]
            r1 = CARD_RANK[opp1]
            s1 = CARD_SUIT[opp1]

            weight = 1.0
            if opp_disc_idx is not None:
                keep_flop_score = eval5_idx(opp0, opp1, b0, b1, b2)
                max_alt = alt0 if alt0 >= alt_flop_score[opp1] else alt_flop_score[opp1]
                has_draw = (s0 == s1 and flop_suit_counts[s0] >= 2)
                if keep_flop_score < max_alt and not has_draw:
                    weight = 0.08
                else:
                    weight = 1.30
                if opp_disc_rank >= 11 and keep_flop_score < (1 << 20) and not has_draw:
                    if (r0 if r0 >= r1 else r1) < opp_disc_rank:
                        weight *= 0.15
                if s0 == opp_disc_suit and s1 == opp_disc_suit:
                    weight *= 0.30

            opp_score = eval7_idx((opp0, opp1, b0, b1, b2, b3, b4))

            if opp_range_bias > 0.05:
                if opp_score >= (2 << 20):
                    weight *= (1.0 + opp_range_bias * 2.0)
                elif opp_score >= (1 << 20):
                    weight *= (1.0 + opp_range_bias * 1.1)
                else:
                    weight *= max(0.30, 1.0 - opp_range_bias * 0.7)
            elif opp_range_bias < -0.05:
                if opp_score < (1 << 20) | (10 << 12):
                    weight *= (1.0 + abs(opp_range_bias) * 1.8)

            total_weight += weight
            if my_score > opp_score:
                weighted_wins += weight
            elif my_score == opp_score:
                weighted_wins += 0.5 * weight

    return weighted_wins / max(total_weight, 1e-9)


def post_discard_equity(hole2_strs, board_strs, my_discard_str, opp_discard_str,
                        opp_range_bias=0.0, num_rollouts=180):
    """
    On the River (5 board cards): Uses `exact_river_equity` (zero-variance combinatorial solver!).
    On the Turn (4 board cards): Uses pre-evaluated flop discard lookup (`alt_flop_score`) for 3x faster rollouts!
    """
    hole_idxs = [CARD_IDX[c] for c in hole2_strs]
    board_idxs = [CARD_IDX[c] for c in board_strs]
    dead_idxs = set(hole_idxs + board_idxs)
    if my_discard_str:
        dead_idxs.add(CARD_IDX[my_discard_str])
    opp_disc_idx = CARD_IDX[opp_discard_str] if opp_discard_str else None
    if opp_disc_idx is not None:
        dead_idxs.add(opp_disc_idx)

    rem_deck = [i for i in range(52) if i not in dead_idxs]

    if len(board_idxs) == 5:
        return exact_river_equity(hole_idxs, board_idxs, rem_deck, opp_disc_idx, opp_range_bias)

    b0, b1, b2, b3 = board_idxs[0], board_idxs[1], board_idxs[2], board_idxs[3]
    opp_disc_rank = CARD_RANK[opp_disc_idx] if opp_disc_idx is not None else 2
    opp_disc_suit = CARD_SUIT[opp_disc_idx] if opp_disc_idx is not None else -1

    alt_flop_score = {}
    if opp_disc_idx is not None:
        for c in rem_deck:
            alt_flop_score[c] = eval5_idx(c, opp_disc_idx, b0, b1, b2)

    flop_suit_counts = [0, 0, 0, 0]
    flop_suit_counts[CARD_SUIT[b0]] += 1
    flop_suit_counts[CARD_SUIT[b1]] += 1
    flop_suit_counts[CARD_SUIT[b2]] += 1

    total_weight = 0.0
    weighted_wins = 0.0

    for _ in range(num_rollouts):
        opp0, opp1, river = random.sample(rem_deck, 3)
        weight = 1.0
        if opp_disc_idx is not None:
            keep_flop_score = eval5_idx(opp0, opp1, b0, b1, b2)
            max_alt = alt_flop_score[opp0] if alt_flop_score[opp0] >= alt_flop_score[opp1] else alt_flop_score[opp1]
            s0, s1 = CARD_SUIT[opp0], CARD_SUIT[opp1]
            has_draw = (s0 == s1 and flop_suit_counts[s0] >= 2)
            if keep_flop_score < max_alt and not has_draw:
                weight = 0.08
            else:
                weight = 1.30
            if opp_disc_rank >= 11 and keep_flop_score < (1 << 20) and not has_draw:
                if max(CARD_RANK[opp0], CARD_RANK[opp1]) < opp_disc_rank:
                    weight *= 0.15
            if s0 == opp_disc_suit and s1 == opp_disc_suit:
                weight *= 0.30

        my_score = eval7_idx((hole_idxs[0], hole_idxs[1], b0, b1, b2, b3, river))
        opp_score = eval7_idx((opp0, opp1, b0, b1, b2, b3, river))

        if opp_range_bias > 0.05:
            if opp_score >= (2 << 20):
                weight *= (1.0 + opp_range_bias * 2.0)
            elif opp_score >= (1 << 20):
                weight *= (1.0 + opp_range_bias * 1.1)
            else:
                weight *= max(0.30, 1.0 - opp_range_bias * 0.7)
        elif opp_range_bias < -0.05:
            if opp_score < (1 << 20) | (10 << 12):
                weight *= (1.0 + abs(opp_range_bias) * 1.8)

        total_weight += weight
        if my_score > opp_score:
            weighted_wins += weight
        elif my_score == opp_score:
            weighted_wins += 0.5 * weight

    return weighted_wins / max(total_weight, 1e-9)
