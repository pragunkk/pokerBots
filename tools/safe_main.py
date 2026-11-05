# ---------------------------------------------------------------------------
# Crash-proof main loop. Paste this over the old main() at the bottom of
# Starter_Kit/adaptive_bot.py, then rebuild with: python tools/build_bot.py
# ---------------------------------------------------------------------------
import traceback

_ERRORS = 0


def _log_error(where):
    """Print the traceback to stderr (never stdout), at most 20 times per match (stderr is capped at 64 KB)."""
    global _ERRORS
    _ERRORS += 1
    if _ERRORS <= 20:
        sys.stderr.write(f"[bot error #{_ERRORS} in {where}]\n{traceback.format_exc()}\n")
        sys.stderr.flush()


def fallback_action(req):
    """Safest legal action: check if free, otherwise fold (call if it is tiny)."""
    legal = req.get("legal_actions") or []
    if "check" in legal:
        return {"type": "action", "action": "check"}
    if "call" in legal and req.get("to_call", 99) <= 2:
        return {"type": "action", "action": "call"}
    return {"type": "action", "action": "fold" if "fold" in legal else "call"}


def legalize(reply, req):
    """Make sure the reply is legal; clamp raise sizes instead of letting the engine turn them into check/fold."""
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
        sys.exit(0)  # engine is gone; nothing left to do


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
                _log_error(t)  # a bookkeeping error must never cost us the match
        # unknown message types (and "notice") are ignored, as the protocol requires


if __name__ == "__main__":
    main()
