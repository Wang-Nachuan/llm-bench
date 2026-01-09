import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any
import csv
from pathlib import Path

import httpx
import logging
from transformers import AutoTokenizer
from config import ClientConfig


@dataclass
class Query:
    qid: int
    arrival_time: float
    input_len: int
    output_len: int


def load_queries(path: str) -> List[Query]:
    queries: List[Query] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        qid = 0
        for row in reader:
            if not row:
                continue
            arrived_at = row.get("arrived_at")
            prefill = row.get("num_prefill_tokens")
            decode = row.get("num_decode_tokens")
            if arrived_at is None or prefill is None or decode is None:
                continue
            arrived_at_s = str(arrived_at).strip()
            prefill_s = str(prefill).strip()
            decode_s = str(decode).strip()
            if not arrived_at_s or not prefill_s or not decode_s:
                continue
            queries.append(
                Query(
                    qid=qid,
                    arrival_time=float(arrived_at_s),
                    input_len=int(prefill_s),
                    output_len=int(decode_s),
                )
            )
            qid += 1
    # Just to be safe
    queries.sort(key=lambda q: q.arrival_time)
    return queries

_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}
_ONE_TOKEN_STRING_CACHE: Dict[str, str] = {}
_ONE_TOKEN_ID_CACHE: Dict[str, int] = {}

def _get_tokenizer_for_model_name(model_name: str) -> AutoTokenizer:
    tok = _TOKENIZER_CACHE.get(model_name)
    if tok is not None:
        return tok
    model_dir = f"/opt/models/{model_name}"
    tok = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    _TOKENIZER_CACHE[model_name] = tok
    return tok

def _find_stable_single_token_string(tok: AutoTokenizer) -> str:
    cache_key = getattr(tok, "name_or_path", "unknown")
    cached = _ONE_TOKEN_STRING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    # Fast candidate set: leading-space words commonly single-token in LLaMA SP tokenizer
    candidates = [
        " the", " a", " to", " in", " on", " is", " and", " of", " for", " with",
        " x", " y", " z", " data", " model", " token", " test"
    ]
    for s in candidates:
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            # Verify round-trip stability
            tid = ids[0]
            dec = tok.decode([tid], clean_up_tokenization_spaces=False)
            re_ids = tok.encode(dec, add_special_tokens=False)
            if len(re_ids) == 1 and re_ids[0] == tid:
                _ONE_TOKEN_STRING_CACHE[cache_key] = dec
                _ONE_TOKEN_ID_CACHE[cache_key] = tid
                # Optional independence sanity check (quick):
                # decode multiple same ids and ensure length scales linearly
                test_ids = [tid] * 8
                test_text = tok.decode(test_ids, clean_up_tokenization_spaces=False)
                if len(tok.encode(test_text, add_special_tokens=False)) != 8:
                    continue
                return dec
    # If none of the small candidates work, raise for explicit fix rather than scanning whole vocab
    raise RuntimeError("No single-token candidate matched from predefined set; extend candidates for this tokenizer.")

def build_prompt_exact_tokens(model_name: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    tok = _get_tokenizer_for_model_name(model_name)
    unit = _find_stable_single_token_string(tok)
    cache_key = getattr(tok, "name_or_path", "unknown")
    unit_id = _ONE_TOKEN_ID_CACHE.get(cache_key)
    if unit_id is None:
        unit_id = tok.encode(unit, add_special_tokens=False)[0]
        _ONE_TOKEN_ID_CACHE[cache_key] = unit_id
    ids = [unit_id] * target_tokens
    prompt = tok.decode(ids, clean_up_tokenization_spaces=False)
    # Verify
    if len(tok.encode(prompt, add_special_tokens=False)) != target_tokens:
        raise RuntimeError("Prompt tokenization mismatch with requested input_len.")
    return prompt

def warm_token_unit(model_name: str) -> None:
    tok = _get_tokenizer_for_model_name(model_name)
    unit = _find_stable_single_token_string(tok)
    cache_key = getattr(tok, "name_or_path", "unknown")
    if cache_key not in _ONE_TOKEN_ID_CACHE:
        _ONE_TOKEN_ID_CACHE[cache_key] = tok.encode(unit, add_special_tokens=False)[0]


async def run_query(
    client: httpx.AsyncClient,
    q: Query,
    base_url: str,
    model_name: str,
    t0: float,
):
    # Build a prompt with EXACTLY q.input_len tokens under the server's tokenizer
    prompt = build_prompt_exact_tokens(model_name, q.input_len)

    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": q.output_len,
        "min_tokens": q.output_len,   # vLLM sampling param
        "ignore_eos": True,           # ensure EOS doesn't cut us short
        "temperature": 0.0,
        "stream": True,
        # Ask server to include usage in the final streamed message if supported
        "stream_options": {"include_usage": True},
    }

    send_time = time.monotonic()
    ttft_time = None
    prev_token_time = None
    per_token_latencies: List[float] = []
    end_time = None
    status_code = None
    error_text = ""
    usage_prompt_tokens = None
    usage_completion_tokens = None
    try:
        url = f"{base_url}/completions"
        async with client.stream("POST", url, json=payload, timeout=None) as resp:
            status_code = resp.status_code
            if status_code != 200:
                error_text = (await resp.aread()).decode("utf-8", errors="ignore")[:200]
                logging.error(f"Query {q.qid} failed: HTTP {status_code} {error_text}")
            else:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        end_time = time.monotonic()
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    now = time.monotonic()
                    u = obj.get("usage")
                    if isinstance(u, dict):
                        usage_prompt_tokens = u.get("prompt_tokens", usage_prompt_tokens)
                        usage_completion_tokens = u.get("completion_tokens", usage_completion_tokens)
                    choices = obj.get("choices")
                    text = ""
                    if isinstance(choices, list) and choices:
                        text = choices[0].get("text") or ""
                    if text:
                        if ttft_time is None:
                            ttft_time = now
                        if prev_token_time is not None:
                            per_token_latencies.append(now - prev_token_time)
                        prev_token_time = now
            if end_time is None:
                end_time = time.monotonic()
    except Exception as e:
        status_code = -1
        error_text = str(e)[:200]
        logging.error(f"Query {q.qid} exception: {error_text}")
        end_time = time.monotonic()

    ttft_s = (ttft_time - send_time) if ttft_time is not None else None
    e2e_s = (end_time - send_time) if end_time is not None else None

    # Temporary verification (uncomment to use)
    # _tok = _get_tokenizer_for_model_name(model_name)
    # _in_len = len(_tok.encode(prompt, add_special_tokens=False))
    # _out_len = len(per_token_latencies)
    # if usage_prompt_tokens is not None and (usage_prompt_tokens != q.input_len and usage_prompt_tokens != q.input_len + 1):
    #     print(f"[VERIFY] prompt_tokens mismatch: expected={q.input_len} usage={usage_prompt_tokens} computed={_in_len}")
    # if usage_completion_tokens is not None and (usage_completion_tokens != q.output_len and usage_completion_tokens != q.output_len + 1):
    #     print(f"[VERIFY] completion_tokens mismatch: expected={q.output_len} usage={usage_completion_tokens} computed={_out_len}")

    return {
        "id": q.qid,
        "status": status_code if status_code is not None else -1,
        "ttft_s": ttft_s,
        "e2e_s": e2e_s,
        "tpot_samples": per_token_latencies,
        "error": error_text,
    }


async def run_benchmark(args):
    queries = load_queries(args.queries)
    if not queries:
        print("No queries loaded.")
        return

    base_url = args.base_url.rstrip("/")
    model_name = args.model_name
    # Pre-warm tokenizer/unit token to avoid per-task discovery work
    warm_token_unit(model_name)

    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        pending: set[asyncio.Task] = set()
        results: List[Dict[str, Any]] = []
        fatal = False
        i = 0
        # Spin scheduler: launch queries at arrival times, and drain finished tasks continuously
        while i < len(queries) or pending:
            # Launch ready queries if not in fatal state
            if not fatal and i < len(queries):
                now_rel = time.monotonic() - t0
                q = queries[i]
                if now_rel >= q.arrival_time:
                    pending.add(asyncio.create_task(run_query(client, q, base_url, model_name, t0)))
                    i += 1
            # Drain any completed tasks without blocking
            if pending:
                done, pending = await asyncio.wait(pending, timeout=0.0, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    r = await d
                    results.append(r)
                    # Abort the current trace on server error or client exception
                    if r.get("status", 200) == 500 or r.get("status", 0) == -1:
                        fatal = True
                        err = r.get("error", "")
                        logging.error(f"Aborting trace due to error on query {r.get('id')}: HTTP {r.get('status')} {err}")
            # If fatal, cancel any remaining pending tasks and exit the loop
            if fatal:
                if pending:
                    for t in pending:
                        t.cancel()
                    try:
                        await asyncio.gather(*pending, return_exceptions=True)
                    except Exception:
                        pass
                    pending.clear()
                break
            # Yield control to progress I/O
            if i < len(queries) and not fatal:
                await asyncio.sleep(0)
        # If not fatal and any tasks remain, ensure they are awaited
        if not fatal and pending:
            results.extend(await asyncio.gather(*pending))
    wall = time.monotonic() - t0

    # Aggregate stats
    ok = sum(1 for r in results if r["status"] == 200)
    total = len(results)
    ttfts = [r["ttft_s"] for r in results if r["status"] == 200 and r["ttft_s"] is not None]
    e2es = [r["e2e_s"] for r in results if r["status"] == 200 and r["e2e_s"] is not None]
    tpot_all: List[float] = []
    for r in results:
        if r["status"] == 200 and r.get("tpot_samples"):
            tpot_all.extend(r["tpot_samples"])
    errors = [r["error"] for r in results if r["status"] != 200 and r.get("error")]

    total_output_tokens = sum(q.output_len for q in queries)
    tps = total_output_tokens / wall if wall > 0 else 0.0
    qps = total / wall if wall > 0 else 0.0

    print(f"Requests: {ok}/{total} succeeded")
    if ttfts:
        print(f"TTFT p50: {sorted(ttfts)[int(0.5 * len(ttfts))]:.3f}s")
        print(f"TTFT p90: {sorted(ttfts)[int(0.9 * len(ttfts))]:.3f}s")
        print(f"TTFT p99: {sorted(ttfts)[int(0.99 * len(ttfts))]:.3f}s")
    if e2es:
        print(f"E2E p50: {sorted(e2es)[int(0.5 * len(e2es))]:.3f}s")
        print(f"E2E p90: {sorted(e2es)[int(0.9 * len(e2es))]:.3f}s")
        print(f"E2E p99: {sorted(e2es)[int(0.99 * len(e2es))]:.3f}s")
    if tpot_all:
        print(f"TPOT p50: {sorted(tpot_all)[int(0.5 * len(tpot_all))]:.3f}s")
        print(f"TPOT p90: {sorted(tpot_all)[int(0.9 * len(tpot_all))]:.3f}s")
        print(f"TPOT p99: {sorted(tpot_all)[int(0.99 * len(tpot_all))]:.3f}s")
    print(f"TPS: {tps:.1f} tok/s")
    print(f"QPS: {qps:.2f} que/s")

    # Return raw metrics for higher-level orchestration if desired
    return {
        "ok": ok,
        "total": total,
        "ttfts": ttfts,
        "e2es": e2es,
        "tpot_samples": tpot_all,
        "wall_time_s": wall,
        "total_output_tokens": total_output_tokens,
        "tps": tps,
        "qps": qps,
        "errors": errors[:10],  # keep it short
    }


def run_benchmark_with_config(cfg: ClientConfig) -> None:
    # Bridge into the existing coroutine-based runner using a Namespace
    ns = argparse.Namespace(
        queries=cfg.queries_path,
        base_url=cfg.base_url,
        model_name=cfg.model_name,
    )
    asyncio.run(run_benchmark(ns))

def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"p50": None, "p90": None, "p99": None}
    s = sorted(values)
    def pct(p: float) -> float:
        idx = int(p * (len(s) - 1))
        return s[idx]
    return {"p50": pct(0.50), "p90": pct(0.90), "p99": pct(0.99)}

def _write_cdf_csv(values: List[float], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("percentile,value_s\n")
        if not values:
            for p in range(0, 101):
                f.write(f"{p},\n")
            return
        s = sorted(values)
        for p in range(0, 101):
            idx = int(p / 100 * (len(s) - 1))
            f.write(f"{p},{s[idx]:.9f}\n")

def run_trace_collect_and_write(cfg: ClientConfig, out_dir: Path) -> Dict[str, Any]:
    # Run one trace and write CDFs + summary to out_dir
    ns = argparse.Namespace(
        queries=cfg.queries_path,
        base_url=cfg.base_url,
        model_name=cfg.model_name,
    )
    metrics = asyncio.run(run_benchmark(ns))
    # Write CDF CSVs
    _write_cdf_csv(metrics["ttfts"], out_dir / "cdf_ttft.csv")
    _write_cdf_csv(metrics["tpot_samples"], out_dir / "cdf_tpot.csv")
    _write_cdf_csv(metrics["e2es"], out_dir / "cdf_e2e.csv")
    # Summary JSON
    summary = {
        "ttft": _percentiles(metrics["ttfts"]),
        "tpot": _percentiles(metrics["tpot_samples"]),
        "e2e": _percentiles(metrics["e2es"]),
        "tps": metrics["tps"],
        "qps": metrics["qps"],
        "requests_ok": metrics["ok"],
        "requests_total": metrics["total"],
        "errors": metrics.get("errors", []),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, indent=2))
    return summary
