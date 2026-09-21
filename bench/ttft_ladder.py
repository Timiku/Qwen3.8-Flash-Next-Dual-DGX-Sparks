#!/usr/bin/env python3
"""TTFT ladder for the KV-offload arm: cold prefill vs parked-restore, cross-boot.

park  (boot 1): for each size, first-visit marker session = cold-prefill TTFT
                + reference answer; waits for the async store to drain.
probe (boot 2): per size, a decoy filler (cold TTFT #2; must answer the decoy,
                proving no parked-tree hit) then the marker re-send = restore
                TTFT (must reproduce boot 1's reference bit-exact).
Rows append to logs/ttft-ladder.jsonl; refs to logs/ttft-ladder-refs.json.
"""
import json
import pathlib
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
SIZES = [32768, 65536, 118000, 236000]          # target prompt tokens
GEN_TOKENS = 400
PW = "GLM-7X4K9-QW-2291"
DECOY = "DECOY-88Z3-MOTH-0517"
STORE_RATE = 2400                                # tok/s measured on the 120k gate row

ROOT = pathlib.Path.home() / "Qwen3.8-Flash-Next-Dual-DGX-Sparks"
LOG = ROOT / "logs" / "ttft-ladder.jsonl"
REFS = ROOT / "logs" / "ttft-ladder-refs.json"


def model_id():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def stream_ttft(messages):
    body = json.dumps({"model": model_id(), "messages": messages,
                       "max_tokens": GEN_TOKENS, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    text = []
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or [{}]
            delta = (choices[0].get("delta") or {})
            delta = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if ttft is None and delta:
                ttft = (time.time() - t0) * 1000
            if delta:
                text.append(delta)
    return {"ttft_ms": ttft, "total_s": round(time.time() - t0, 2),
            "text": "".join(text),
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


def build_prompt(n_target, password, salt=0):
    para = (f"Ledger run stamp {salt}. "
            f"The vault archive holds the maintenance ledger for the north wing. "
            f"Entries are filed by week; each week's page carries the duty roster, "
            f"the delivery manifests, and the seal count. The archive password is "
            f"{password}. Clerks repeat it aloud once per shift, never twice. ")
    filler = ("The east stairwell drains into the cistern; its grates are lifted "
              "on the first Monday of each month and scrubbed before the rains "
              "return. manifests seal ledger roster rains grates cistern stairwell ")
    words = []
    # password paragraph roughly once per ~600 words keeps it distributed
    block = para + filler * 9
    block_words = len(block.split())
    reps = max(1, int(n_target * 0.72 / block_words))
    body = " ".join([block] * reps)
    q = ("\n\nQuestion: what is the archive password? "
         "Answer with only the password, nothing else.\nAnswer:")
    return body + q


def append(row):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))


def park():
    salt = int(time.time())
    refs = {"salt": salt}
    for n in SIZES:
        p = build_prompt(n, PW, salt)
        r = stream_ttft([{"role": "user", "content": p}])
        refs[str(n)] = {"ttft_ms": r["ttft_ms"], "prompt_tokens": r["prompt_tokens"],
                        "ref": r["text"]}
        append({"stage": "park", "target_n": n, "ttft_ms": r["ttft_ms"],
                "total_s": r["total_s"], "prompt_tokens": r["prompt_tokens"],
                "answer": r["text"][:120],
                "store_wait_s": round(n / STORE_RATE + 20, 1)})
        REFS.write_text(json.dumps(refs, indent=1), encoding="utf-8")
        time.sleep(n / STORE_RATE + 20)   # async store drain before the next size


def probe():
    refs = json.loads(REFS.read_text(encoding="utf-8"))
    salt = refs.get("salt", 0)
    for n in SIZES:
        if str(n) not in refs:
            append({"stage": "probe", "target_n": n, "skipped": "no ref from park"})
            continue
        decoy = stream_ttft([{"role": "user", "content": build_prompt(n, DECOY, salt)}])
        got = stream_ttft([{"role": "user", "content": build_prompt(n, PW, salt)}])
        ref = refs[str(n)]["ref"]
        append({"stage": "probe", "target_n": n,
                "prompt_tokens": got["prompt_tokens"],
                "cold_ttft_ms": decoy["ttft_ms"],
                "decoy_ok": DECOY in decoy["text"],
                "restore_ttft_ms": got["ttft_ms"],
                "restore_total_s": got["total_s"],
                "marker_text_match": got["text"] == ref,
                "ref": ref[:120], "got": got["text"][:120]})


if __name__ == "__main__":
    {"park": park, "probe": probe}[sys.argv[1]]()
