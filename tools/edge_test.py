"""Drive a bot process through protocol edge cases and check every reply. Usage: protocol_edge.py BOT"""
import json, subprocess, sys, time, threading, queue
BOT = sys.argv[1]
fails = []

class Bot:
    def __init__(self):
        self.p = subprocess.Popen([sys.executable, "-I", BOT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.err = []
        threading.Thread(target=lambda: self.err.extend(self.p.stderr), daemon=True).start()
    def _rd(self):
        for line in self.p.stdout: self.q.put(line)
        self.q.put(None)
    def send(self, obj):
        self.p.stdin.write((obj if isinstance(obj, str) else json.dumps(obj)) + "\n"); self.p.stdin.flush()
    def reply(self, timeout=5.0):
        t = time.time()
        try: line = self.q.get(timeout=timeout)
        except queue.Empty: return None, timeout
        return line, time.time() - t
    def no_reply(self, wait=0.3):
        try: line = self.q.get(timeout=wait); return line
        except queue.Empty: return None

def check_action(name, b, req, max_ms=1000):
    b.send(req)
    line, dt = b.reply()
    if line is None: fails.append(f"{name}: NO REPLY"); return None
    try: r = json.loads(line)
    except ValueError: fails.append(f"{name}: not JSON: {line!r}"); return None
    if r.get("type") != "action" or r.get("action") not in req["legal_actions"]:
        fails.append(f"{name}: illegal {r} legal={req['legal_actions']}"); return r
    if r["action"] == "raise" and not (isinstance(r.get("amount"), int) and req["min_raise"] <= r["amount"] <= req["max_raise"]):
        fails.append(f"{name}: bad raise amount {r}")
    if dt * 1000 > max_ms: fails.append(f"{name}: slow {dt*1000:.0f} ms")
    print(f"  ok  {name:55s} -> {r}  ({dt*1000:.0f} ms)")
    return r

def check_discard(name, b, req):
    b.send(req)
    line, dt = b.reply()
    if line is None: fails.append(f"{name}: NO REPLY"); return None
    r = json.loads(line)
    if r.get("type") != "discard" or r.get("card") not in req["hole_cards"]: fails.append(f"{name}: bad discard {r}")
    print(f"  ok  {name:55s} -> {r}  ({dt*1000:.0f} ms)")
    return r

def AR(street, board, pot, my_bet, opp_bet, my_stack, opp_stack, legal, hist, raises_left=3, mn=None, mx=None, t=30000, hid=1):
    d = {"type": "action_request", "hand_id": hid, "street": street, "board": board, "pot": pot, "my_bet": my_bet,
         "opp_bet": opp_bet, "to_call": opp_bet - my_bet, "my_stack": my_stack, "opp_stack": opp_stack,
         "legal_actions": legal, "raises_left": raises_left, "history": hist, "time_left_ms": t}
    if "raise" in legal: d["min_raise"], d["max_raise"] = mn, mx
    return d
H = lambda st, a, act, amt=None: dict({"street": st, "actor": a, "action": act}, **({"amount": amt} if amt is not None else {}))
INIT = {"type": "init", "protocol_version": "1.0", "total_hands": 200, "starting_stack": 200, "small_blind": 1, "big_blind": 2, "time_bank_ms": 30000, "per_response_ms": 5000}

b = Bot()
t0 = time.time(); b.send(INIT)
print("1. Preflop shove, we call, all-in discard, showdown")
b.send({"type": "new_hand", "hand_id": 1, "seat": "big_blind", "hole_cards": ["As", "Ah", "7c"], "stack": 200, "opp_stack": 200})
check_action("BB facing button shove (no raise legal)", b, AR("preflop", [], 202, 2, 200, 198, 0, ["fold", "call"], [H("preflop", "opp", "raise", 200)], raises_left=2))
check_discard("discard after all-in, no flop action", b, {"type": "discard_request", "hand_id": 1, "hole_cards": ["As", "Ah", "7c"], "board": ["2d", "9s", "Kc"], "pot": 400, "my_stack": 0, "opp_stack": 0, "time_left_ms": 29000})
b.send({"type": "discard_result", "hand_id": 1, "my_discard": "7c", "opp_discard": "3d", "hole_cards": ["As", "Ah"]})
b.send({"type": "hand_result", "hand_id": 1, "result": "showdown", "net": 200, "board": ["2d", "9s", "Kc", "4h", "Jd"], "my_hole_cards": ["As", "Ah"], "opp_hole_cards": ["Kd", "Qd"], "opp_discard": "3d", "history": [H("preflop", "opp", "raise", 200), H("preflop", "me", "call", 200)]})
if b.no_reply(): fails.append("replied to a non-request")

print("2. Short all-in raise only (min_raise == max_raise), cap reached, opp all-in with raises_left > 0")
b.send({"type": "new_hand", "hand_id": 2, "seat": "button", "hole_cards": ["Ks", "Kh", "2c"], "stack": 200, "opp_stack": 200})
hist = [H("preflop", "me", "raise", 6), H("preflop", "opp", "raise", 20), H("preflop", "me", "raise", 60), H("preflop", "opp", "raise", 180)]
check_action("facing 4th raise preflop: cap reached, raises_left 0", b, AR("preflop", [], 240, 60, 180, 140, 20, ["fold", "call"], hist, raises_left=0))
b.send({"type": "hand_result", "hand_id": 2, "result": "fold", "net": -60, "board": [], "my_hole_cards": ["Ks", "Kh", "2c"], "opp_hole_cards": None, "opp_discard": None, "history": hist + [H("preflop", "me", "fold")]})
b.send({"type": "new_hand", "hand_id": 3, "seat": "button", "hole_cards": ["9s", "8s", "7d"], "stack": 200, "opp_stack": 200})
check_action("button open, normal", b, AR("preflop", [], 3, 1, 2, 199, 198, ["fold", "call", "raise"], [], mn=4, mx=200))
hist = [H("preflop", "me", "raise", 6), H("preflop", "opp", "call", 6), H("flop", "opp", "check")]
check_action("flop, raise legal but only all-in (min==max==3)", b, AR("flop", ["Ts", "6s", "2h"], 12, 0, 0, 3, 194, ["check", "raise"], hist, raises_left=4, mn=3, mx=3))
hist2 = hist + [H("flop", "me", "check")]
check_discard("discard 3-card after check-check", b, {"type": "discard_request", "hand_id": 3, "hole_cards": ["9s", "8s", "7d"], "board": ["Ts", "6s", "2h"], "pot": 12, "my_stack": 194, "opp_stack": 194, "time_left_ms": 29000})
b.send({"type": "discard_result", "hand_id": 3, "my_discard": "7d", "opp_discard": "Ac", "hole_cards": ["9s", "8s"]})
check_action("turn: opp all-in, raise absent though raises_left 4", b, AR("turn", ["Ts", "6s", "2h", "Jd"], 206, 0, 194, 194, 0, ["fold", "call"], hist2 + [H("turn", "opp", "raise", 194)], raises_left=4))
b.send({"type": "hand_result", "hand_id": 3, "result": "fold", "net": -6, "board": ["Ts", "6s", "2h", "Jd"], "my_hole_cards": ["9s", "8s"], "opp_hole_cards": None, "opp_discard": "Ac", "history": hist2})

print("3. Junk and unknown input between requests")
for junk in ["", "   ", "not json", "[1,2,3]", "42", '{"no_type":1}', '{"type":"notice","code":"illegal_action","applied":"fold"}', '{"type":"future_message","x":1}', '{"type":"hand_result"}', '{"type":"discard_result"}', '{"type":"new_hand"}']:
    b.send(junk)
if b.no_reply(0.5): fails.append("replied to junk")
b.send({"type": "new_hand", "hand_id": 4, "seat": "big_blind", "hole_cards": ["Qc", "Jc", "3h"], "stack": 200, "opp_stack": 200})
check_action("request after junk + extra unknown fields", b, dict(AR("preflop", [], 8, 2, 6, 198, 194, ["fold", "call", "raise"], [H("preflop", "opp", "raise", 6)], mn=10, mx=200), extra_field={"a": 1}))
r = AR("preflop", [], 8, 2, 6, 198, 194, ["fold", "call", "raise"], [H("preflop", "opp", "raise", 6)], mn=10, mx=200); del r["history"]; del r["max_raise"]
check_action("request missing history and max_raise", b, r)
r = AR("flop", ["2c", "7h", "Kd"], 16, 0, 0, 192, 192, ["check", "raise"], [], mn=2, mx=192); r["board"] = ["2c", "7h", "XX"]
check_action("request with an invalid card on the board", b, r)

print("4. Tiny time bank")
b.send({"type": "new_hand", "hand_id": 5, "seat": "button", "hole_cards": ["Ad", "Kd", "Qd"], "stack": 200, "opp_stack": 200})
check_action("preflop with 40 ms left", b, AR("preflop", [], 3, 1, 2, 199, 198, ["fold", "call", "raise"], [], mn=4, mx=200, t=40), max_ms=200)
check_action("flop with 40 ms left", b, AR("flop", ["Ac", "Kc", "5h"], 12, 0, 0, 0, 194, ["check", "raise"], [H("preflop", "me", "raise", 6), H("preflop", "opp", "call", 6), H("flop", "opp", "check")], raises_left=4, mn=2, mx=194, t=40), max_ms=200)
check_discard("discard with 40 ms left", b, {"type": "discard_request", "hand_id": 5, "hole_cards": ["Ad", "Kd", "Qd"], "board": ["Ac", "Kc", "5h"], "pot": 12, "my_stack": 194, "opp_stack": 194, "time_left_ms": 40})
b.send({"type": "discard_result", "hand_id": 5, "my_discard": "Qd", "opp_discard": "2s", "hole_cards": ["Ad", "Kd"]})
check_action("river overbet facing, 40 ms left", b, AR("river", ["Ac", "Kc", "5h", "9c", "3c"], 400, 0, 194, 194, 0, ["fold", "call"], [H("preflop", "me", "raise", 6), H("preflop", "opp", "call", 6), H("flop", "opp", "check"), H("flop", "me", "check"), H("turn", "opp", "check"), H("turn", "me", "check"), H("river", "opp", "raise", 194)], raises_left=4, t=40), max_ms=300)
b.send({"type": "hand_result", "hand_id": 5, "result": "showdown", "net": 6, "board": ["Ac", "Kc", "5h", "9c", "3c"], "my_hole_cards": ["Ad", "Kd"], "opp_hole_cards": ["8c", "2c"], "opp_discard": "2s", "history": []})

print("5. Hands beyond total_hands, then match_end exits cleanly")
for hid in range(6, 260):
    b.send({"type": "new_hand", "hand_id": hid, "seat": "button" if hid % 2 else "big_blind", "hole_cards": ["2c", "7d", "9h"], "stack": 200, "opp_stack": 200})
    b.send({"type": "hand_result", "hand_id": hid, "result": "fold", "net": -1, "board": [], "my_hole_cards": ["2c", "7d", "9h"], "opp_hole_cards": None, "opp_discard": None, "history": [H("preflop", "me", "fold")]})
check_action("request at hand 260 of a 200-hand leg", b, AR("preflop", [], 3, 1, 2, 199, 198, ["fold", "call", "raise"], [], mn=4, mx=200, hid=260))
b.send({"type": "match_end", "reason": "complete"})
try:
    code = b.p.wait(timeout=3); print(f"  exit code after match_end: {code}")
    if code != 0: fails.append(f"exit code {code}")
except subprocess.TimeoutExpired:
    fails.append("did not exit after match_end"); b.p.kill()
err = "".join(b.err)
print(f"  stderr bytes: {len(err)}")
if err: print("  stderr head:", err[:600])

print("6. EOF without match_end")
b2 = Bot(); b2.send(INIT); b2.p.stdin.close()
try: print("  exit code on EOF:", b2.p.wait(timeout=3))
except subprocess.TimeoutExpired: fails.append("hangs on EOF"); b2.p.kill()

print("\nFAILURES:" if fails else "\nALL EDGE CASES PASSED")
for f in fails: print("  -", f)
