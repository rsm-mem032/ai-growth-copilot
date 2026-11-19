"""Step 2: LLM classification into structured JSON.
Setup: pip install anthropic ; export ANTHROPIC_API_KEY=... ; optional MODEL=<model id>
Run:   python classify.py 20         -> trial run on the first 20 signals
       python classify.py            -> all signals -> predictions.jsonl
"""
import os, sys, json, anthropic

# The SDK retries rate-limit (429) and server errors with exponential backoff.
client = anthropic.Anthropic(max_retries=6)
MODEL = os.environ.get("MODEL", "claude-sonnet-5-5")
TYPES = {"user_intent", "creator_lead", "partnership_fit", "irrelevant"}
ACTIONS = {"draft_reply", "draft_dm", "shortlist", "ignore"}
SYSTEM = """You help the overseas growth team of an AI creative platform find opportunities in public posts.
Classify the signal and return ONLY a JSON object with keys:
type: one of "user_intent" (wants to make/solve something the product helps with), "creator_lead" (a creator worth supporting/featuring),
      "partnership_fit" (channel/community with distribution potential), "irrelevant"
score: integer 0-100 (priority for the team)
why_this_matters: one sentence
recommended_action: one of "draft_reply", "draft_dm", "shortlist", "ignore"
draft: a short, non-spammy draft message if action is draft_reply or draft_dm, else ""
Never claim to be a human. Be conservative: use "irrelevant" when unsure."""

def call(payload):
    msg = client.messages.create(model=MODEL, max_tokens=500, system=SYSTEM,
                                 messages=[{"role": "user", "content": json.dumps(payload)}])
    t = msg.content[0].text
    pred = json.loads(t[t.find("{"): t.rfind("}") + 1])
    if pred.get("type") not in TYPES or pred.get("recommended_action") not in ACTIONS:
        raise ValueError(f"invalid label in model output: {t[:200]}")
    pred["score"] = max(0, min(100, int(pred.get("score", 0))))
    return pred

def classify(sig):
    payload = {k: sig.get(k) for k in ("platform", "kind", "text", "author", "channel_subs")}
    try:
        return call(payload)
    except (ValueError, TypeError):  # malformed JSON / labels: retry once
        return call(payload)

def main(inp="signals.jsonl", out="predictions.jsonl", limit=None):
    n = errors = 0
    with open(inp) as f, open(out, "w") as g:
        for line in f:
            if limit is not None and n >= limit:
                break
            sig = json.loads(line)
            try:
                pred = classify(sig)
            except Exception as e:
                errors += 1
                pred = {"type": "error", "score": 0, "why_this_matters": str(e), "recommended_action": "ignore", "draft": ""}
            g.write(json.dumps({**sig, **{f"pred_{k}": v for k, v in pred.items()}}) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  {n} classified ({errors} errors)")
    print(f"done -> {out}: {n} signals, {errors} errors")

if __name__ == "__main__":
    main(limit=int(sys.argv[1]) if len(sys.argv) > 1 else None)
