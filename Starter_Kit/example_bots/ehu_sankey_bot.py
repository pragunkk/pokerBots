"""
Proven Algorithm 2: University of Alberta (Poki/Loki) Effective Hand Strength (EHS & EHS^2) Bot.
Computes:
  - Current Hand Strength (HS): probability our hand currently beats opponent's hand
  - Positive Potential (Ppot): probability a currently behind hand improves to win by river
  - Negative Potential (Npot): probability a currently ahead hand gets outdrawn by river
  - Linear EHS = HS * (1 - Npot) + (1 - HS) * Ppot (used for Calling decisions)
  - Aggressive EHS^2 = HS^2 * (1 - Npot) + (1 - HS^2) * Ppot (used for Betting/Raising decisions,
    naturally favoring high-upside nut draws and monsters over vulnerable 1-pair hands!)
"""
import os, sys, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fast_eval import (
    CARD_IDX,
    CARD_SUIT,
    eval5_idx,
    eval7_idx,
    hole_interaction_level,
    preflop_3card_strength,
)


def compute_ehs_metrics(hole_idxs, board_idxs, dead_idxs, num_samples=75):
    """
    Returns (hs, ppot, npot, ehs, ehs2) for a 2-card holding on Flop or Turn (or exact HS on River).
    """
    rem = [i for i in range(52) if i not in dead_idxs]
    to_come = 5 - len(board_idxs)

    if to_come == 0:
        # River: Ppot = Npot = 0
        wins = 0.0
        my_s = eval7_idx(hole_idxs + board_idxs)
        for _ in range(num_samples):
            o0, o1 = random.sample(rem, 2)
            op_s = eval7_idx([o0, o1] + board_idxs)
            if my_s > op_s:
                wins += 1.0
            elif my_s == op_s:
                wins += 0.5
        hs = wins / num_samples
        return hs, 0.0, 0.0, hs, hs * hs

    # Flop or Turn: track Ahead/Tied/Behind now vs Win/Tie/Loss at River
    ahead_cnt, behind_cnt = 0.0, 0.0
    ahead_lost, behind_won = 0.0, 0.0
    cur_wins = 0.0

    for _ in range(num_samples):
        samp = random.sample(rem, 2 + to_come)
        o0, o1 = samp[0], samp[1]
        fut_board = board_idxs + samp[2:]

        # Current street evaluation
        if len(board_idxs) == 3:
            my_now = eval5_idx(hole_idxs[0], hole_idxs[1], board_idxs[0], board_idxs[1], board_idxs[2])
            op_now = eval5_idx(o0, o1, board_idxs[0], board_idxs[1], board_idxs[2])
        else:
            # 4-card turn board: evaluate best 5 of 6 via eval7_idx with dummy duplicate handled or eval_cards
            # Fast proxy: compare 5-card subset using board[:3] + turn
            my_now = max(
                eval5_idx(hole_idxs[0], hole_idxs[1], board_idxs[0], board_idxs[1], board_idxs[3]),
                eval5_idx(hole_idxs[0], hole_idxs[1], board_idxs[0], board_idxs[2], board_idxs[3]),
                eval5_idx(hole_idxs[0], hole_idxs[1], board_idxs[1], board_idxs[2], board_idxs[3]),
                eval5_idx(hole_idxs[0], hole_idxs[1], board_idxs[0], board_idxs[1], board_idxs[2]),
            )
            op_now = max(
                eval5_idx(o0, o1, board_idxs[0], board_idxs[1], board_idxs[3]),
                eval5_idx(o0, o1, board_idxs[0], board_idxs[2], board_idxs[3]),
                eval5_idx(o0, o1, board_idxs[1], board_idxs[2], board_idxs[3]),
                eval5_idx(o0, o1, board_idxs[0], board_idxs[1], board_idxs[2]),
            )

        my_riv = eval7_idx(hole_idxs + fut_board)
        op_riv = eval7_idx([o0, o1] + fut_board)

        riv_score = 1.0 if my_riv > op_riv else (0.5 if my_riv == op_riv else 0.0)
        if my_now > op_now:
            cur_wins += 1.0
            ahead_cnt += 1.0
            ahead_lost += (1.0 - riv_score)
        elif my_now == op_now:
            cur_wins += 0.5
            ahead_cnt += 0.5
            behind_cnt += 0.5
            behind_won += 0.5 * riv_score
            ahead_lost += 0.5 * (1.0 - riv_score)
        else:
            behind_cnt += 1.0
            behind_won += riv_score

    hs = cur_wins / num_samples
    ppot = behind_won / max(1.0, behind_cnt)
    npot = ahead_lost / max(1.0, ahead_cnt)
    ehs = hs * (1.0 - npot) + (1.0 - hs) * ppot
    ehs2 = (hs * hs) * (1.0 - npot) + (1.0 - hs * hs) * ppot
    return hs, ppot, npot, ehs, ehs2


class EHUSankeyBot:
    def __init__(self):
        self.hole3 = []
        self.hole2 = []
        self.my_discard = None
        self.opp_discard = None

    def choose_discard_and_eval(self, hole3_strs, flop3_strs):
        h_idxs = [CARD_IDX[c] for c in hole3_strs]
        b_idxs = [CARD_IDX[c] for c in flop3_strs]
        dead = set(h_idxs + b_idxs)
        best_card = hole3_strs[0]
        best_score = -1.0
        best_metrics = None
        for i in range(3):
            disc = hole3_strs[i]
            keep = [h_idxs[j] for j in range(3) if j != i]
            hs, ppot, npot, ehs, ehs2 = compute_ehs_metrics(keep, b_idxs, dead, num_samples=45)
            # Score discard by 0.65 * EHS + 0.35 * EHS^2 (favoring nut draw upside!)
            score = 0.65 * ehs + 0.35 * ehs2
            if score > best_score:
                best_score = score
                best_card = disc
                best_metrics = (hs, ppot, npot, ehs, ehs2)
        return best_card, best_metrics

    def choose_discard(self, req):
        best_card, _ = self.choose_discard_and_eval(req["hole_cards"], req["board"])
        return {"type": "discard", "card": best_card}

    def choose_action(self, req):
        legal = req["legal_actions"]
        street = req["street"]
        to_call = req["to_call"]
        pot = req["pot"]
        opp_bet = req["opp_bet"]

        if street == "preflop":
            s = preflop_3card_strength(self.hole3)
            if opp_bet <= 2:
                if s >= 0.36 and "raise" in legal:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], 6))}
                if to_call == 0 or s >= 0.24:
                    return {"type": "action", "action": "call" if to_call > 0 else "check"}
                return {"type": "action", "action": "fold"}
            else:
                if s >= 0.70 and "raise" in legal and opp_bet <= 25:
                    return {"type": "action", "action": "raise", "amount": max(req["min_raise"], min(req["max_raise"], opp_bet * 3))}
                if s >= 0.45 and opp_bet <= 30:
                    return {"type": "action", "action": "call"}
                if s >= 0.75:
                    return {"type": "action", "action": "call"}
                return {"type": "action", "action": "fold"}

        if street == "flop":
            _, (_, ppot, npot, ehs, ehs2) = self.choose_discard_and_eval(self.hole3, req["board"])
            inter = 2
        else:
            h_idxs = [CARD_IDX[c] for c in self.hole2]
            b_idxs = [CARD_IDX[c] for c in req["board"]]
            dead = set(h_idxs + b_idxs)
            if self.my_discard:
                dead.add(CARD_IDX[self.my_discard])
            if self.opp_discard:
                dead.add(CARD_IDX[self.opp_discard])
            _, ppot, npot, ehs, ehs2 = compute_ehs_metrics(h_idxs, b_idxs, dead, num_samples=65)
            inter = hole_interaction_level(self.hole2, req["board"])

        pot_odds = to_call / max(1, pot + to_call)

        if to_call == 0:
            # Bet using EHS^2 or high Positive Potential (semi-bluffing strong draws)
            if "raise" in legal and (ehs2 >= 0.42 or ehs >= 0.60 or ppot >= 0.26):
                if street == "river" and inter == 0:
                    return {"type": "action", "action": "check"}
                frac = 0.75 if ehs2 >= 0.62 else 0.55
                amt = max(req["min_raise"], min(req["max_raise"], int(pot * frac)))
                return {"type": "action", "action": "raise", "amount": amt}
            return {"type": "action", "action": "check"}
        else:
            if street == "river" and inter == 0 and to_call > pot * 0.20:
                return {"type": "action", "action": "fold"}
            # Raise using EHS^2
            if ehs2 >= 0.64 and inter >= 3 and "raise" in legal:
                amt = max(req["min_raise"], min(req["max_raise"], opp_bet + int((pot + to_call) * 0.75)))
                return {"type": "action", "action": "raise", "amount": amt}
            # Call using linear EHS + Ppot implied odds bonus
            implied_bonus = ppot * 0.15 if street != "river" else 0.0
            if ehs + implied_bonus >= pot_odds + 0.04:
                return {"type": "action", "action": "call"}
            return {"type": "action", "action": "fold"}


def main():
    bot = EHUSankeyBot()
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
