#!/usr/bin/env python3
"""Throw thousands of random (but realistic) requests at the bot and check it never crashes,
never answers illegally and never takes too long.

    python tools/fuzz.py Starter_Kit/bot.py 5000"""
import importlib.util, random, sys, time, traceback

spec = importlib.util.spec_from_file_location("botmod", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
DECK = [r + s for r in "23456789TJQKA" for s in "shdc"]
errors, illegal, worst = {}, [], 0.0

for i in range(N):
    b = m.AdaptiveBot()
    b.handle_init({"type": "init", "total_hands": random.choice([400, 1000]), "time_bank_ms": 60000})
    d = random.sample(DECK, 11)
    b.handle_new_hand({"type": "new_hand", "hand_id": random.randint(1, 400),
                       "seat": random.choice(["button", "big_blind"]), "hole_cards": d[:3],
                       "stack": 200, "opp_stack": 200})
    street = random.choice(["preflop", "flop", "turn", "river", "discard"])
    board = d[6:6 + {"preflop": 0, "flop": 3, "turn": 4, "river": 5, "discard": 3}[street]]
    t0 = time.perf_counter()
    try:
        if street in ("turn", "river"):
            disc = random.choice(d[:3])
            b.handle_discard_result({"type": "discard_result", "my_discard": disc, "opp_discard": d[3],
                                     "hole_cards": [c for c in d[:3] if c != disc]})
        if street == "discard":
            r = b.choose_discard({"type": "discard_request", "hole_cards": d[:3], "board": board,
                                  "pot": random.randint(4, 400), "my_stack": 100, "opp_stack": 100,
                                  "time_left_ms": random.choice([100, 3000, 60000])})
            if r.get("card") not in d[:3]:
                illegal.append(("discard", r))
        else:
            stack = random.randint(0, 198)
            my_bet = random.randint(0, 50)
            opp_bet = my_bet + random.choice([0, 0, random.randint(1, 150)])
            if street == "preflop":
                opp_bet = max(opp_bet, 2)
                if opp_bet == my_bet:
                    my_bet = opp_bet = 2
            to_call = opp_bet - my_bet
            legal = ["fold", "call"] if to_call > 0 else ["check"]
            raises_left = random.randint(0, 4)
            max_r = my_bet + stack
            if raises_left > 0 and max_r > opp_bet:
                legal.append("raise")
            req = {"type": "action_request", "hand_id": 1, "street": street, "board": board,
                   "pot": my_bet + opp_bet + random.randint(3, 200), "my_bet": my_bet, "opp_bet": opp_bet,
                   "to_call": to_call, "my_stack": stack, "opp_stack": random.randint(0, 198),
                   "legal_actions": legal, "raises_left": raises_left, "history": [],
                   "time_left_ms": random.choice([100, 2000, 3000, 60000])}
            if "raise" in legal:
                req["min_raise"] = min(opp_bet + max(2, to_call), max_r)
                req["max_raise"] = max_r
            r = b.choose_action(req)
            a = r.get("action")
            bad = a not in legal or (a == "raise" and not (
                isinstance(r.get("amount"), int) and req["min_raise"] <= r["amount"] <= req["max_raise"]))
            if bad:
                illegal.append((street, legal, r))
    except Exception:
        key = traceback.format_exc().strip().splitlines()[-1]
        errors[key] = errors.get(key, 0) + 1
    worst = max(worst, time.perf_counter() - t0)

print(f"cases {N}   crashes {sum(errors.values())}   illegal replies {len(illegal)}   slowest reply {worst*1000:.1f} ms")
for k, v in errors.items():
    print(f"  CRASH x{v}: {k}")
for x in illegal[:5]:
    print("  ILLEGAL:", x)
sys.exit(1 if errors or illegal else 0)
