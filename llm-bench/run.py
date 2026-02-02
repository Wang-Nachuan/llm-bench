#!/usr/bin/env python
import subprocess
import sys
import time
from pathlib import Path
import os
import shutil
from types import SimpleNamespace
import logging
import threading
import re

import httpx
from config import default_orchestrator_config
from server import launch_server
from client import run_trace_collect_and_write
try:
    import torch
except Exception:
    torch = None

def wait_for_server(base_url: str, server_proc: subprocess.Popen) -> None:
    """Block until /v1/models is reachable, or raise if the server process exits."""
    while True:
        # If the server process died, fail fast with a useful error.
        rc = server_proc.poll()
        if rc is not None:
            raise RuntimeError(
                f"vLLM server exited during startup (exit_code={rc}). "
            )
        try:
            resp = httpx.get(f"{base_url}/models", timeout=2.0)
            if resp.status_code == 200:
                logging.info("Server is ready.")
                return
        except Exception:
            pass
        time.sleep(2.0)


def discover_traces(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.csv") if p.is_file()])

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)

def write_gpu_topo(results_root: Path) -> None:
    """Run `nvidia-smi topo -m` and `nvidia-smi` and write a clean text file."""
    results_root.mkdir(parents=True, exist_ok=True)
    out_path = results_root / "gpu_topo.txt"

    def run_cmd(argv: list[str]) -> tuple[int, str]:
        p = subprocess.run(argv, capture_output=True, text=True)
        combined = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, _strip_ansi(combined)

    rc_topo, topo_out = run_cmd(["nvidia-smi", "topo", "-m"])
    rc_smi, smi_out = run_cmd(["nvidia-smi"])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("### nvidia-smi topo -m\n")
        f.write(f"(exit_code={rc_topo})\n")
        f.write(topo_out.rstrip() + "\n\n")
        f.write("### nvidia-smi\n")
        f.write(f"(exit_code={rc_smi})\n")
        f.write(smi_out.rstrip() + "\n")


def check_system(cfg) -> None:
    if torch is None:
        raise RuntimeError("PyTorch not available; cannot verify GPU configuration.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this container/runtime.")
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 0:
        raise RuntimeError("No CUDA GPUs detected.")
    gpu_names = [torch.cuda.get_device_properties(i).name for i in range(num_gpus)]
    expected = cfg.server.gpu_type.lower()
    def matches(name: str) -> bool:
        return expected in name.lower()
    if not all(matches(n) for n in gpu_names):
        raise RuntimeError(f"GPU type mismatch. Expected '{expected}', found: {gpu_names}")
    tp, pp, dp = cfg.server.tp_size, cfg.server.pp_size, cfg.server.dp_size
    world = tp * pp * dp
    if world > num_gpus:
        raise RuntimeError(f"Parallelism ranks exceed available GPUs: TP*PP*DP={world} > {num_gpus}")
    logging.info(f"Detected GPUs: {gpu_names}")
    logging.info(f"Parallelism: TP={tp}, PP={pp}, DP={dp}, world={world} (num_gpus={num_gpus})")


class PowerSampler(threading.Thread):
    def __init__(self, sample_period_s: float) -> None:
        super().__init__(daemon=True)
        self.sample_period_s = sample_period_s
        self._stop_event = threading.Event()
        self._start_t = None
        self.samples = {}  # gpu_index -> list[(t_rel_s, watts)]
        self.num_gpus = 0

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        self._start_t = time.monotonic()
        while not self._stop_event.is_set():
            t_now = time.monotonic()
            t_rel = t_now - self._start_t
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    stderr=subprocess.STDOUT,
                    timeout=5,
                ).decode("utf-8", errors="ignore")
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                self.num_gpus = max(self.num_gpus, len(lines))
                for idx, ln in enumerate(lines):
                    try:
                        val = float(ln)
                    except Exception:
                        continue
                    self.samples.setdefault(idx, []).append((t_rel, val))
            except Exception:
                # Best-effort sampling; ignore transient errors
                pass
            time.sleep(self.sample_period_s)


def write_power_csv(samples: dict, agg_period_s: float, out_csv: Path) -> None:
    # Determine duration
    max_t = 0.0
    for vals in samples.values():
        if vals:
            max_t = max(max_t, vals[-1][0])
    if max_t <= 0:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("time_s,avg_power_w\n")
        return
    num_bins = int(max_t / agg_period_s) + 1
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("time_s,avg_power_w\n")
        # For each bin, average ALL GPU samples falling within the window
        for b in range(num_bins):
            w_start = b * agg_period_s
            w_end = min((b + 1) * agg_period_s, max_t)
            all_vals = []
            for _, vals in samples.items():
                if not vals:
                    continue
                for (t, v) in vals:
                    if (t >= w_start and t < w_end) or (b == num_bins - 1 and t <= w_end):
                        all_vals.append(v)
            if not all_vals:
                continue
            avg = sum(all_vals) / len(all_vals)
            f.write(f"{w_start:.3f},{avg:.3f}\n")


def main():
    cfg = default_orchestrator_config()

    # Output roots + logging
    results_root = Path(cfg.client.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    log_path = results_root / "bench.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(str(log_path)),
        ],
    )

    # Capture GPU topology/status early into a clean text file.
    write_gpu_topo(results_root)

    # Resolve trace search root (default: queries/dev_<TP*PP*DP>/, test mode: queries/test/)
    base_path = Path(cfg.client.queries_path)
    if os.getenv("TEST_MODE", "0") == "1":
        search_root = base_path / "test"
        logging.info("TEST_MODE=1: using traces under queries/test/")
    else:
        world = cfg.server.tp_size * cfg.server.pp_size * cfg.server.dp_size
        if world not in (8, 16):
            raise RuntimeError(f"Unsupported device count (TP*PP*DP)={world}; expected 8 or 16.")
        search_root = base_path / f"dev_{world}"
    logging.info(f"Searching traces under: {search_root}")

    traces = discover_traces(search_root)
    if not traces:
        raise FileNotFoundError(f"No CSV traces found under {search_root}")

    # Suppress verbose HTTP client logs to avoid per-request noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.info("Checking system configurations...")
    check_system(cfg)

    # Prepare a persistent 4K backup config so we can swap config.json per trace.
    model_dir = Path(cfg.server.model_dir)
    config_json = model_dir / "config.json"
    config_4k = model_dir / "config_4k.json"
    if not config_4k.exists():
        shutil.copyfile(config_json, config_4k)

    try:
        logging.info(f"Running {len(traces)} trace(s) sequentially...")

        for idx, trace in enumerate(traces):
            rel_name = trace.stem  # filename without extension
            out_dir = results_root / rel_name
            out_dir.mkdir(parents=True, exist_ok=True)
            os.environ["VLLM_LOG_FILE"] = str(out_dir / "vllm.log")
            ns_cfg = SimpleNamespace(queries_path=str(trace),
                                     base_url=cfg.client.base_url,
                                     model_name=cfg.client.model_name)
            logging.info(f"Start trace: {trace}")
            status = "ok"
            error_note = ""

            # Select per-trace model config + max_model_len
            is_reasoning = rel_name.startswith("reasoning")
            cfg.server.max_model_len = 16384 + 2048 if is_reasoning else 4096
            chosen_cfg = model_dir / ("config_16k.json" if is_reasoning else "config_4k.json")
            shutil.copyfile(chosen_cfg, config_json)

            # Always restart server per trace to isolate performance and avoid CUDA state leakage.
            server_proc = launch_server(cfg.server)
            logging.info("Waiting for server to become ready...")
            wait_for_server(cfg.client.base_url, server_proc)

            # Start GPU power sampler
            ps = PowerSampler(sample_period_s=cfg.client.power_sample_period_s)
            ps.start()
            try:
                summary = run_trace_collect_and_write(ns_cfg, out_dir)
                errs_list = summary.get("errors", [])
                errs = " ".join(errs_list).lower()
                # If any query error occurred in this trace, mark error
                if errs_list:
                    status = "error"
                    error_note = "Query errors detected (see summary/errors)."
                # Highlight OOM specifically if present
                if "out of memory" in errs or "oom" in errs:
                    status = "error"
                    error_note = "Detected OOM in vLLM responses."
            except Exception as e:
                status = "error"
                error_note = f"Exception during trace: {e}"
                with open(out_dir / "error.txt", "w", encoding="utf-8") as f:
                    f.write(error_note)
            # Stop sampler and write power CSV
            try:
                ps.stop()
                ps.join(timeout=5.0)
                write_power_csv(ps.samples, cfg.client.power_agg_period_s, out_dir / "gpu_power.csv")
            except Exception as e:
                logging.info(f"Failed to write GPU power CSV: {e}")
            logging.info(f"End trace: {trace} status={status} {error_note}")
            # Stop server after each trace (always)
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.warning("Server did not exit in time; killing.")
                server_proc.kill()

        logging.info("All traces completed.")
    finally:
        # Ensure log file is flushed
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
