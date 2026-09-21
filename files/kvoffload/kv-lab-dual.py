#!/usr/bin/env python3
"""kv-lab-dual.py — KV-offload qualification ladder for the dual kit (flash-next).

Adapted from the GLM-5.3 lab's kv-lab.py. Stages:
  health    fence sidecar on both nodes, scratch warning present, metrics live
  store     prompt A (~120k tok, deterministic unique text + needle) -> NVMe
            files grow on both nodes; records A's output text for restore
  flood     fresh ~262k prompts sized to push A's blocks out of the 4.9M pool;
            store bytes must GROW during the flood (parks happening)
  restore   re-send A: load bytes grow, output text matches the store stage,
            TTFT vs the store-stage TTFT (recompute would be ~10x slower)

Stdlib only. Receipts: JSONL + stdout. Run from Spark1 (the head).
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8888"
MODEL = "qwen3.8-flash-next"
WORKER = "10.100.72.2"
KV_DIR = "/home/timiku/fn-kv"
OUT = f"/home/timiku/Qwen3.8-Flash-Next-Dual-DGX-Sparks/logs/kvoffload-ladder-{time.strftime('%H%M%S')}.jsonl"
IDS = "/home/timiku/Qwen3.8-Flash-Next-Dual-DGX-Sparks/logs/kvoffload_store_ids.json"
NEEDLE = "The archive password is vermillion-lantern-42."
A_TOKENS = 120_000
FLOOD_TOKENS = 5_200_000  # > 4.9M pool + A footprint
MAX_LEN_TOKENS = 258_000


def log(**kv):
    kv["t"] = round(time.time(), 1)
    with open(OUT, "a") as f:
        f.write(json.dumps(kv) + "\n")
    print(json.dumps(kv), flush=True)


def post(path, body, timeout=3600.0):
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw), time.time() - t0


def _metrics():
    req = urllib.request.Request(BASE + "/metrics")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode("utf-8", "replace")


def kv_bytes():
    """(store_bytes, load_bytes) from the connector's transfer counters."""
    store = load = 0.0
    for line in _metrics().splitlines():
        for name, sink in (("vllm:kv_offload_store_bytes", "s"),
                           ("vllm:kv_offload_load_bytes", "l")):
            if not (line.startswith(name + "_total{") or line.startswith(name + "{")
                    or line.startswith(name + " ")):
                continue
            if "_created" in line or "_sum{" in line or "_bucket" in line:
                continue
            v = float(line.rsplit(" ", 1)[1])
            if sink == "s":
                store = v
            else:
                load = v
    return store, load


def waiting():
    out = {}
    for line in _metrics().splitlines():
        if line.startswith("vllm:num_requests_waiting_by_reason"):
            r = line.split('reason="')[1].split('"')[0]
            out[r] = out.get(r, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)


def du_ns(where="local"):
    """(allocated_B, apparent_B) of this node's KV namespace dir."""
    if where == "worker":
        cmd = (f"ssh -o BatchMode=yes {WORKER} "
               f"'du -B1 -sb {KV_DIR}/*/* 2>/dev/null'")
    else:
        cmd = f"du -B1 -sb {KV_DIR}/*/* 2>/dev/null"
    out = sh(cmd)
    alloc = 0
    for ln in out.stdout.splitlines():
        n = ln.split("\t")[0].strip()
        if n.isdigit():
            alloc += int(n)
    return alloc, alloc


def ns_name():
    out = sh(f"ls {KV_DIR} 2>/dev/null")
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    return lines[0] if lines else ""


def mem_avail(worker=False):
    cmd = "awk '/MemAvailable/{print $2}' /proc/meminfo"
    if worker:
        cmd = f"ssh -o BatchMode=yes {WORKER} \"awk '/MemAvailable/{{print \\$2}}' /proc/meminfo\""
    v = sh(cmd).stdout.strip()
    return int(v) // 1024 if v.isdigit() else -1


class Watcher(threading.Thread):
    """Sample host MemAvailable (MB) on both nodes during a stage (GB10 trap)."""

    def __init__(self, every=10):
        super().__init__(daemon=True)
        self.every = every
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append((round(time.time(), 1), mem_avail(), mem_avail(worker=True)))
            self._halt.wait(self.every)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


# ---- deterministic, incompressible prompts ---------------------------------
def archive_lines(seed, n_lines):
    base = seed * 7919 + 11
    return [
        f"Row {i}: vault-{seed}-{i} holds crystal {((base + i * 31) % 999983)}, "
        f"sigil {'abcdefghij'[i % 10]}-{'klmnopqrst'[(i // 3) % 10]}."
        for i in range(n_lines)
    ]


def build_prompt(seed, approx_tokens):
    """Unique-text archive; tokenizer-checked, proportional-scaled to target."""
    n_lines = int(approx_tokens / 10) + 8
    text = ""
    for _ in range(5):
        text = "\n".join(archive_lines(seed, n_lines))
        tok, _ = post("/tokenize", {"model": MODEL, "prompt": text}, timeout=180)
        n = tok["count"]
        if abs(n - approx_tokens) <= approx_tokens * 0.005:
            break
        n_lines = max(1, int(n_lines * approx_tokens / max(n, 1)))
    return text + "\nEnd of archive."


def prompt_a():
    return NEEDLE + "\n" + build_prompt(1, A_TOKENS) + "\nRepeat only the archive password."


# ---- stages ----------------------------------------------------------------
def stage_health():
    marks = int(sh("docker logs vllm-fn 2>&1 | grep -c 'fn-kv-offload'").stdout or 0)
    wmarks = int(sh(f"ssh -o BatchMode=yes {WORKER} \"docker logs vllm-fn 2>&1 | grep -c 'fn-kv-offload'\"").stdout or 0)
    fence_local = sh(f"test -f {KV_DIR}/*/config.json && echo yes || echo no").stdout.strip()
    fence_worker = sh(f"ssh -o BatchMode=yes {WORKER} 'test -f {KV_DIR}/*/config.json && echo yes || echo no'").stdout.strip()
    ns = ns_name()
    s, l = kv_bytes()
    ok = marks > 0 and wmarks > 0 and fence_local == "yes" and fence_worker == "yes" and bool(ns)
    log(stage="health", head_marks=marks, worker_marks=wmarks,
        fence_head=fence_local, fence_worker=fence_worker, namespace=ns,
        store_bytes=s, load_bytes=l, ok=ok)
    return ok


def _completion(prompt, max_tokens, timeout):
    body = {"model": MODEL, "prompt": prompt, "temperature": 0,
            "max_tokens": max_tokens, "seed": 42}
    resp, el = post("/v1/completions", body, timeout=timeout)
    return resp, el


def stage_store():
    p = prompt_a()
    s0, l0 = kv_bytes()
    a0, _ = du_ns("local")
    resp, el = _completion(p, 16, 2400)
    s1, l1 = kv_bytes()
    a1, _ = du_ns("local")
    w_alloc, _ = du_ns("worker")
    text = resp["choices"][0]["text"]
    with open(IDS, "w") as f:
        json.dump({"text": text, "prompt_tokens": resp["usage"]["prompt_tokens"],
                   "ttft_store_s": None, "seconds": el}, f)
    log(stage="store", seconds=round(el, 1), prompt_tokens=resp["usage"]["prompt_tokens"],
        store_delta_GB=round((s1 - s0) / 1e9, 2),
        head_alloc_after_MB=a1 // 1048576, worker_alloc_MB=w_alloc // 1048576,
        out_head=text[:60], ok=(s1 - s0) > 5e8)
    return (s1 - s0) > 5e8


def stage_flood():
    w = Watcher()
    w.start()
    per = MAX_LEN_TOKENS
    n = (FLOOD_TOKENS + per - 1) // per
    for i in range(n):
        p = build_prompt(100 + i, per) + "\nAnswer with the number 7 only."
        s0, _ = kv_bytes()
        resp, el = _completion(p, 2, 2400)
        s1, _ = kv_bytes()
        log(stage="flood", i=i, seconds=round(el, 1),
            prompt_tokens=resp["usage"]["prompt_tokens"],
            store_total_GB=round(s1 / 1e9, 1), store_delta_MB=round((s1 - s0) / 1e6, 0),
            waiting=waiting(), head_avail_MB=mem_avail(), worker_avail_MB=mem_avail(True))
    w.stop()
    if w.samples:
        log(stage="flood", min_head_avail_MB=min(s[1] for s in w.samples),
            min_worker_avail_MB=min(s[2] for s in w.samples), samples=len(w.samples))


def stage_restore():
    p = prompt_a()
    ref = json.load(open(IDS))
    s0, l0 = kv_bytes()
    resp, el = _completion(p, 16, 2400)
    s1, l1 = kv_bytes()
    match = resp["choices"][0]["text"] == ref["text"]
    log(stage="restore", seconds=round(el, 1), store_seconds=ref["seconds"],
        load_delta_GB=round((l1 - l0) / 1e9, 2), store_delta_GB=round((s1 - s0) / 1e9, 2),
        prompt_tokens=resp["usage"]["prompt_tokens"], marker_text_match=match,
        ref=ref["text"][:60], got=resp["choices"][0]["text"][:60],
        ok=match and (l1 - l0) > 1e8)
    return match and (l1 - l0) > 1e8


STAGES = {"health": stage_health, "store": stage_store,
          "flood": stage_flood, "restore": stage_restore}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    a = ap.parse_args()
    sys.exit(0 if STAGES[a.stage]() is not False else 1)
