"""
Ultra-fast Hand Evaluator with Startup O(1) Lookup Tables (POW5 + Flush Bitmask),
Precomputed 1,755 Canonical 3-Card Preflop Evaluator, Soft Sigmoid Preflop Range Weighting,
Post-Flop Discard Optimizer, Exact Zero-Variance Combinatorial River Solver,
Pre-Evaluated Turn Rollout Engine, and Board Texture Danger Analyzer (v8 Championship Engine).
Uses only Python 3.11 standard library.
"""

import math
import random
from collections import Counter
from itertools import combinations, combinations_with_replacement
from preflop_table import PREFLOP_TABLE_1755

RANKS = "23456789TJQKA"
SUITS = "shdc"
RANK_VAL = {r: i + 2 for i, r in enumerate(RANKS)}
SUIT_VAL = {s: i for i, s in enumerate(SUITS)}
ALL_CARDS = tuple(r + s for r in RANKS for s in SUITS)
CARD_IDX = {c: i for i, c in enumerate(ALL_CARDS)}

CARD_RANK = tuple(RANK_VAL[c[0]] for c in ALL_CARDS)
CARD_SUIT = tuple(SUIT_VAL[c[1]] for c in ALL_CARDS)
CARD_BIT = tuple(1 << (i // 4) for i in range(52))

POW5 = tuple(5 ** i for i in range(13))
CARD_POW5 = tuple(POW5[i // 4] for i in range(52))
SUITADD = tuple(1 << (4 * (i & 3)) for i in range(52))

COMBOS_7_5 = tuple(combinations(range(7), 5))
COMBOS_6_5 = tuple(combinations(range(6), 5))

NON_FLUSH_LUT = {}
FLUSH_BIT_LUT = {}
N7 = {}
N6 = {}
FL = {}
FSUIT = [-1] * 65536
_TABLES_READY = False


def _eval5_from_ranks(r0, r1, r2, r3, r4, s_flush):
    """Exact 5-card score in the canonical bit layout (cat << 20) | tiebreakers (ranks 2..14)."""
    if r0 < r1: r0, r1 = r1, r0
    if r3 < r4: r3, r4 = r4, r3
    if r2 < r4: r2, r4 = r4, r2
    if r2 < r3: r2, r3 = r3, r2
    if r0 < r3: r0, r3 = r3, r0
    if r0 < r2: r0, r2 = r2, r0
    if r1 < r4: r1, r4 = r4, r1
    if r1 < r3: r1, r3 = r3, r1
    if r1 < r2: r1, r2 = r2, r1

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

    if r0 == r3:
        return (7 << 20) | (r0 << 4) | r4
    if r1 == r4:
        return (7 << 20) | (r1 << 4) | r0

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


def init_fast_tables():
    """Precomputes O(1) 5-, 6-, and 7-card lookup tables during the free startup window."""
    global _TABLES_READY
    if _TABLES_READY:
        return
    for r0 in range(13):
        p0 = POW5[r0]
        for r1 in range(r0, 13):
            p1 = p0 + POW5[r1]
            for r2 in range(r1, 13):
                p2 = p1 + POW5[r2]
                for r3 in range(r2, 13):
                    p3 = p2 + POW5[r3]
                    for r4 in range(r3, 13):
                        if r0 == r4:
                            continue
                        key = p3 + POW5[r4]
                        NON_FLUSH_LUT[key] = _eval5_from_ranks(r0 + 2, r1 + 2, r2 + 2, r3 + 2, r4 + 2, False)
                        if r0 < r1 < r2 < r3 < r4:
                            mask = (1 << r0) | (1 << r1) | (1 << r2) | (1 << r3) | (1 << r4)
                            FLUSH_BIT_LUT[mask] = _eval5_from_ranks(r0 + 2, r1 + 2, r2 + 2, r3 + 2, r4 + 2, True)

    nf = NON_FLUSH_LUT
    for combo in combinations_with_replacement(range(13), 7):
        if combo[0] == combo[4] or combo[1] == combo[5] or combo[2] == combo[6]:
            continue
        p = [POW5[r] for r in combo]
        best = 0
        for i0, i1, i2, i3, i4 in COMBOS_7_5:
            v = nf[p[i0] + p[i1] + p[i2] + p[i3] + p[i4]]
            if v > best:
                best = v
        N7[sum(p)] = best

    for combo in combinations_with_replacement(range(13), 6):
        if combo[0] == combo[4] or combo[1] == combo[5]:
            continue
        p = [POW5[r] for r in combo]
        best = 0
        for i0, i1, i2, i3, i4 in COMBOS_6_5:
            v = nf[p[i0] + p[i1] + p[i2] + p[i3] + p[i4]]
            if v > best:
                best = v
        N6[sum(p)] = best

    fb = FLUSH_BIT_LUT
    for mask in range(8192):
        bits = [i for i in range(13) if (mask >> i) & 1]
        if len(bits) < 5:
            continue
        best = 0
        for sub in combinations(bits, 5):
            v = fb[(1 << sub[0]) | (1 << sub[1]) | (1 << sub[2]) | (1 << sub[3]) | (1 << sub[4])]
            if v > best:
                best = v
        FL[mask] = best

    for s in range(65536):
        for su in range(4):
            if ((s >> (4 * su)) & 15) >= 5:
                FSUIT[s] = su
                break

    _TABLES_READY = True


init_fast_tables()


def eval5_idx(c0, c1, c2, c3, c4):
    """O(1) 5-card evaluator taking 5 integer card indices (0..51)."""
    if CARD_SUIT[c0] == CARD_SUIT[c1] == CARD_SUIT[c2] == CARD_SUIT[c3] == CARD_SUIT[c4]:
        return FLUSH_BIT_LUT[CARD_BIT[c0] | CARD_BIT[c1] | CARD_BIT[c2] | CARD_BIT[c3] | CARD_BIT[c4]]
    return NON_FLUSH_LUT[CARD_POW5[c0] + CARD_POW5[c1] + CARD_POW5[c2] + CARD_POW5[c3] + CARD_POW5[c4]]


def eval6_idx(cards6):
    """O(1) 6-card evaluator taking 6 integer card indices (0..51)."""
    a, b, c, d, e, f = cards6
    v = N6[CARD_POW5[a] + CARD_POW5[b] + CARD_POW5[c] + CARD_POW5[d] + CARD_POW5[e] + CARD_POW5[f]]
    fs = FSUIT[SUITADD[a] + SUITADD[b] + SUITADD[c] + SUITADD[d] + SUITADD[e] + SUITADD[f]]
    if fs >= 0:
        m = 0
        for x in cards6:
            if (x & 3) == fs:
                m |= CARD_BIT[x]
        fv = FL[m]
        if fv > v:
            v = fv
    return v


def eval7_idx(cards7):
    """O(1) Best 5-card hand score out of 7 card indices."""
    a, b, c, d, e, f, g = cards7
    v = N7[CARD_POW5[a] + CARD_POW5[b] + CARD_POW5[c] + CARD_POW5[d] + CARD_POW5[e] + CARD_POW5[f] + CARD_POW5[g]]
    fs = FSUIT[SUITADD[a] + SUITADD[b] + SUITADD[c] + SUITADD[d] + SUITADD[e] + SUITADD[f] + SUITADD[g]]
    if fs >= 0:
        m = 0
        for x in cards7:
            if (x & 3) == fs:
                m |= CARD_BIT[x]
        fv = FL[m]
        if fv > v:
            v = fv
    return v


def eval_cards(hole_strs, board_strs):
    """Evaluate best 5-card hand from string cards (hole + board, total 5, 6, or 7)."""
    idxs = [CARD_IDX[c] for c in hole_strs] + [CARD_IDX[c] for c in board_strs]
    n = len(idxs)
    if n == 7:
        return eval7_idx(idxs)
    elif n == 6:
        return eval6_idx(idxs)
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


_PRE_STRENGTH_CACHE = {}
_STRENGTH_SORTED = []


def pre_strength_idx(a, b, c):
    """Preflop table strength of a 3-card hand given as card indices (cached)."""
    key = (a, b, c) if a < b < c else tuple(sorted((a, b, c)))
    v = _PRE_STRENGTH_CACHE.get(key)
    if v is None:
        v = preflop_3card_strength([ALL_CARDS[key[0]], ALL_CARDS[key[1]], ALL_CARDS[key[2]]])
        _PRE_STRENGTH_CACHE[key] = v
    return v


def strength_quantile(q):
    """Strength s such that a fraction q of all 22,100 starting hands is weaker than s."""
    if not _STRENGTH_SORTED:
        _STRENGTH_SORTED.extend(sorted(preflop_3card_strength(list(c)) for c in combinations(ALL_CARDS, 3)))
    q = min(0.999, max(0.0, q))
    return _STRENGTH_SORTED[int(q * (len(_STRENGTH_SORTED) - 1))]


class RangeWeight:
    """Soft preflop range: how likely an opponent holding a 3-card hand of strength s is to have
    played the way they did. `lo` = soft floor of the range (weaker hands fold), `hi` = soft cap
    (stronger hands would have raised). Never returns 0, so a surprising hand is only discounted."""

    __slots__ = ("lo", "hi", "width", "floor", "cap_floor", "_memo")

    def __init__(self, lo=None, hi=None, width=0.045, floor=0.12, cap_floor=0.40):
        self.lo, self.hi, self.width, self.floor, self.cap_floor = lo, hi, width, floor, cap_floor
        self._memo = {}

    def __call__(self, s):
        w = self._memo.get(s)
        if w is None:
            w = 1.0
            if self.lo is not None:
                z = (s - self.lo) / self.width
                w = self.floor + (1.0 - self.floor) / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            if self.hi is not None:
                z = (s - self.hi) / self.width
                w *= 1.0 - (1.0 - self.cap_floor) / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            self._memo[s] = w
        return w


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
    """Returns True if hole_strs + board_strs has a 4-flush draw (9 outs) or 8-out straight draw (OESD / double-gutshot)."""
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
    if 14 in hole_r:
        hole_r = hole_r | {1}
    # 1. Open-ended straight draw (4 consecutive ranks with both ends open: 2 <= low and high <= 13)
    for i in range(len(ranks) - 3):
        if ranks[i + 3] - ranks[i] == 3 and ranks[i] >= 2 and ranks[i + 3] <= 13:
            if any(ranks[i] <= hr <= ranks[i + 3] for hr in hole_r):
                return True
    # 2. Double-gutshot (two distinct missing inside ranks that complete a 5-card straight)
    gutshot_outs = set()
    rank_set = set(ranks)
    for low in range(1, 11):
        present = [r for r in range(low, low + 5) if r in rank_set]
        if len(present) == 4 and any(low <= hr <= low + 4 for hr in hole_r):
            missing = (set(range(low, low + 5)) - set(present)).pop()
            gutshot_outs.add(missing)
    if len(gutshot_outs) >= 2:
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


def evaluate_discard_options(hole3_strs, flop3_strs, num_rollouts=220, opp_vpip=0.65, pre_w=None):
    """
    Evaluates all 3 flop discard choices, combining:
      1. Expected Showdown Equity against opponent's VPIP/RangeWeight-filtered range and optimal discard
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
        if pre_w is not None:
            w = pre_w(pre_strength_idx(opp0, opp1, opp2))
        else:
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


def exact_river_equity(hole_idxs, board_idxs, rem_deck, opp_disc_idx, opp_range_bias=0.0, pre_w=None):
    """
    Zero-Variance Combinatorial Enumeration over opponent 2-card hands on the River!
    Precomputes single-card flop-discard scores in 43 calls and my_score in 1 O(1) lookup.
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

            if pre_w is not None and opp_disc_idx is not None:
                weight *= pre_w(pre_strength_idx(opp0, opp1, opp_disc_idx))

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
                        opp_range_bias=0.0, num_rollouts=240, pre_w=None):
    """
    On the River (5 board cards): Uses `exact_river_equity` (zero-variance combinatorial solver!).
    On the Turn (4 board cards): Uses O(1) N7 lookup tables + pre-evaluated flop discard lookup!
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
        return exact_river_equity(hole_idxs, board_idxs, rem_deck, opp_disc_idx, opp_range_bias, pre_w)

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
            if pre_w is not None:
                weight *= pre_w(pre_strength_idx(opp0, opp1, opp_disc_idx))

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
