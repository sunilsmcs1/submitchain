"""LLM-assisted triage (Section 2.5).

Stage 2: each journal summarises ITS OWN manuscript locally into five fields.
Stage 3: the agent compares the two summaries field by field and returns a structured report.
The model is reached through an OpenAI-compatible endpoint (e.g. a local Ollama server), so no
unpublished text leaves the journal's infrastructure. Prompts, schema and model are fixed so both
journals obtain the same report from the same inputs (temperature 0).

    python triage.py summarise manuscript.json summary.json --model <model>
    python triage.py compare summary_a.json summary_b.json report.json --model <model>
    python triage.py evaluate flagged_pairs.jsonl summaries.jsonl out.json --model <model>
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request

FIELDS = ["research_question", "method", "data", "principal_findings", "stated_contributions"]
RATINGS = ["distinct", "related", "substantially_overlapping", "same_work"]

SUMMARY_PROMPT = (
    "Summarise the manuscript below for a duplicate-submission check. Return ONLY a JSON object with "
    "exactly these keys: " + ", ".join(FIELDS) + ". Each value is at most 60 words, factual, in your own words, "
    "with specific numbers, datasets and method names where the manuscript gives them. Do not name the authors "
    "or their institutions.\n\nMANUSCRIPT\nTitle: {title}\nAbstract: {abstract}\nMain text (may be truncated): {body}"
)
COMPARE_PROMPT = (
    "Two journals received the manuscripts summarised below. Decide whether they report the same work. "
    "Return ONLY a JSON object with keys: rating (one of " + ", ".join(RATINGS) + "), "
    "overlapping_fields (list of field names that match in substance), differing_fields (list), and "
    "rationale (at most 80 words). Use same_work when question, method, data and findings all match even if "
    "wording differs; substantially_overlapping when most match but one manuscript adds material results; "
    "related when only the topic or method is shared; distinct otherwise.\n\nSUMMARY A\n{a}\n\nSUMMARY B\n{b}"
)


def chat(base_url: str, model: str, prompt: str, api_key: str = "none") -> str:
    body = json.dumps({"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates code fences and preambles)."""
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("model reply contains no JSON object")
    return json.loads(m.group(0))


def validate_summary(s: dict) -> dict:
    missing = [f for f in FIELDS if not isinstance(s.get(f), str) or not s[f].strip()]
    if missing:
        raise ValueError(f"summary is missing fields: {missing}")
    return {f: s[f].strip() for f in FIELDS}


def validate_report(r: dict, shared_author: bool) -> dict:
    if r.get("rating") not in RATINGS:
        raise ValueError(f"rating must be one of {RATINGS}")
    clean = lambda xs: [x for x in (xs or []) if x in FIELDS]
    return {"rating": r["rating"], "overlapping_fields": clean(r.get("overlapping_fields")),
            "differing_fields": clean(r.get("differing_fields")), "rationale": str(r.get("rationale", ""))[:600],
            "shared_author": bool(shared_author),
            "advisory": "This report supports the editors' judgement. It does not decide the case."}


def summarise(manuscript: dict, ask, max_body_words: int = 3000) -> dict:
    body = " ".join((manuscript.get("body") or "").split()[:max_body_words])
    prompt = SUMMARY_PROMPT.format(title=manuscript["title"], abstract=manuscript["abstract"], body=body)
    for _ in range(2):  # one retry on malformed output
        try:
            return validate_summary(parse_json(ask(prompt)))
        except ValueError:
            continue
    raise ValueError("model did not return a valid summary")


def compare(a: dict, b: dict, ask, shared_author: bool = False) -> dict:
    prompt = COMPARE_PROMPT.format(a=json.dumps(a, indent=1), b=json.dumps(b, indent=1))
    for _ in range(2):
        try:
            return validate_report(parse_json(ask(prompt)), shared_author)
        except ValueError:
            continue
    raise ValueError("model did not return a valid report")


def evaluate(pairs: list[dict], summaries: dict, ask, keep=("substantially_overlapping", "same_work")) -> dict:
    """Ablation of Section 2.7: precision/recall of fingerprint alerts before and after triage.
    pairs: flagged pairs {left_id, right_id, label}; summaries: id -> summary."""
    tp0 = sum(p["label"] for p in pairs); fp0 = len(pairs) - tp0
    tp1 = fp1 = 0; ratings = []
    for p in pairs:
        r = compare(summaries[p["left_id"]], summaries[p["right_id"]], ask)
        ratings.append({**p, "rating": r["rating"]})
        if r["rating"] in keep:
            tp1 += p["label"]; fp1 += 1 - p["label"]
    pr = lambda tp, fp: tp / (tp + fp) if tp + fp else 0.0
    return {"flagged": len(pairs), "precision_before": pr(tp0, fp0), "precision_after": pr(tp1, fp1),
            "duplicates_kept": tp1 / tp0 if tp0 else 0.0, "ratings": ratings}


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("summarise", "compare", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("inputs", nargs="+"); p.add_argument("--model", required=True)
        p.add_argument("--base-url", default="http://localhost:11434/v1"); p.add_argument("--api-key", default="none")
    a = ap.parse_args()
    ask = lambda prompt: chat(a.base_url, a.model, prompt, a.api_key)
    if a.cmd == "summarise":
        src, dst = a.inputs
        json.dump(summarise(json.load(open(src)), ask), open(dst, "w"), indent=1)
    elif a.cmd == "compare":
        sa, sb, dst = a.inputs
        json.dump(compare(json.load(open(sa)), json.load(open(sb)), ask), open(dst, "w"), indent=1)
    else:
        pairs_f, sums_f, dst = a.inputs
        pairs = [json.loads(l) for l in open(pairs_f)]
        sums = {s["id"]: s["summary"] for s in map(json.loads, open(sums_f))}
        json.dump(evaluate(pairs, sums, ask), open(dst, "w"), indent=1)


if __name__ == "__main__":
    main()
