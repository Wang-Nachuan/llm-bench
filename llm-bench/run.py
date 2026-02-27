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
from config import default_orchestrator_config, bench_run_configs
from server import launch_server
from client import run_trace_collect_and_write
from cdf import write_cdf_csv
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
    tp, pp, dp = cfg.server.tp_size, cfg.server.pp_size, cfg.server.dp_size
    world = tp * pp * dp
    if world > num_gpus:
        raise RuntimeError(f"Parallelism ranks exceed available GPUs: TP*PP*DP={world} > {num_gpus}")
    logging.info(f"Detected GPUs: {gpu_names}")
    logging.info(f"Parallelism: TP={tp}, PP={pp}, DP={dp}, world={world} (num_gpus={num_gpus})")


class TelemetrySampler(threading.Thread):
    def __init__(
        self,
        *,
        metrics_url: str,
        model_name: str,
        power_sample_period_s: float,
        batch_sample_period_s: float,
    ) -> None:
        super().__init__(daemon=True)
        self.metrics_url = metrics_url
        self.model_name = model_name
        self.power_sample_period_s = power_sample_period_s
        self.batch_sample_period_s = batch_sample_period_s
        self._stop_event = threading.Event()
        self._start_t = None
        self.power_samples = {}  # gpu_index -> list[(t_rel_s, watts)]
        self.batch_sizes: list[int] = []
        self.exc: Exception | None = None

        self._batch_pat = re.compile(
            r'^vllm:num_requests_running\{[^}]*model_name="'
            + re.escape(model_name)
            + r'"[^}]*\}\s+([0-9.eE+-]+)\s*$',
            re.MULTILINE,
        )

    def stop(self) -> None:
        self._stop_event.set()

    def _sample_power(self, t_rel: float) -> None:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            timeout=5,
        ).decode("utf-8", errors="ignore")
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        for idx, ln in enumerate(lines):
            try:
                val = float(ln)
            except Exception:
                continue
            self.power_samples.setdefault(idx, []).append((t_rel, val))

    def _sample_batch_size(self) -> None:
        resp = httpx.get(self.metrics_url, timeout=2.0)
        resp.raise_for_status()
        m = self._batch_pat.search(resp.text)
        if m:
            self.batch_sizes.append(int(float(m.group(1))))

    def run(self) -> None:
        self._start_t = time.monotonic()
        next_power = self._start_t
        next_batch = self._start_t
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                t_rel = now - self._start_t
                if now >= next_power:
                    try:
                        self._sample_power(t_rel)
                    except Exception:
                        # Best-effort power sampling
                        pass
                    next_power = now + self.power_sample_period_s
                if now >= next_batch:
                    self._sample_batch_size()
                    next_batch = now + self.batch_sample_period_s

                sleep_s = min(next_power, next_batch) - time.monotonic()
                if sleep_s > 0:
                    time.sleep(min(sleep_s, 0.5))
        except Exception as e:
            self.exc = e


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
    base_cfg = default_orchestrator_config()
    bench_runs = bench_run_configs()

    # Output roots + logging
    results_root = Path(base_cfg.client.results_root)
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

    # Suppress verbose HTTP client logs to avoid per-request noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    base_path = Path(base_cfg.client.queries_path)

    try:
        logging.info(f"Running {len(bench_runs)} bench configuration(s)...")

        for bench in bench_runs:
            cfg = default_orchestrator_config()
            for k, v in bench.server_overrides.items():
                setattr(cfg.server, k, v)
            # Keep client model_name consistent with served model name.
            cfg.client.model_name = cfg.server.served_model_name

            bench_root = results_root / bench.name
            bench_root.mkdir(parents=True, exist_ok=True)

            search_root = base_path / bench.trace_folder
            logging.info(f"Bench config={bench.name}: searching traces under {search_root}")
            traces = discover_traces(search_root)
            if not traces:
                raise FileNotFoundError(f"No CSV traces found under {search_root}")

            logging.info("Checking system configurations...")
            check_system(cfg)

            # Prepare a persistent 4K backup config so we can swap config.json per trace.
            model_dir = Path(cfg.server.model_dir)
            config_json = model_dir / "config.json"
            config_4k = model_dir / "config_4k.json"
            if not config_4k.exists():
                shutil.copyfile(config_json, config_4k)

            logging.info(f"Bench config={bench.name}: running {len(traces)} trace(s) sequentially...")
            for idx, trace in enumerate(traces):
                rel_name = trace.stem  # filename without extension
                out_dir = bench_root / rel_name
                out_dir.mkdir(parents=True, exist_ok=True)
                os.environ["VLLM_LOG_FILE"] = str(out_dir / "vllm.log")
                ns_cfg = SimpleNamespace(queries_path=str(trace),
                                         base_url=cfg.client.base_url,
                                         model_name=cfg.client.model_name)
                logging.info(f"Start trace: {trace}")
                status = "ok"
                error_note = ""

                # Select per-trace model config + max_model_len
                extra_server_args = list(bench.extra_server_args)
                is_reasoning = rel_name.startswith("reasoning")
                if is_reasoning:
                    cfg.server.max_model_len = 16384 + 2048
                    cfg.server.max_num_batched_tokens = 3072
                    cfg.server.max_num_seqs = 128
                    # extra_server_args += ["--no-enable-chunked-prefill"]
                    chosen_cfg = model_dir / "config_16k.json"
                else:
                    cfg.server.max_model_len = 4096
                    chosen_cfg = model_dir / "config_4k.json"
                shutil.copyfile(chosen_cfg, config_json)

                # Always restart server per trace to isolate performance and avoid CUDA state leakage.
                server_proc = launch_server(cfg.server, extra_args=extra_server_args)
                logging.info("Waiting for server to become ready...")
                wait_for_server(cfg.client.base_url, server_proc)

                base = cfg.client.base_url.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                metrics_url = f"{base}/metrics"
                ts = TelemetrySampler(
                    metrics_url=metrics_url,
                    model_name=cfg.server.served_model_name,
                    power_sample_period_s=cfg.server.power_sample_period_s,
                    batch_sample_period_s=cfg.server.batch_sample_period_s,
                )
                ts.start()
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
                try:
                    ts.stop()
                    ts.join(timeout=5.0)
                    if ts.exc is not None:
                        raise ts.exc
                    if not ts.batch_sizes:
                        raise RuntimeError(
                            f"No batch samples collected from {metrics_url}. "
                            f"Check metric vllm:num_requests_running with model_name={cfg.server.served_model_name!r}."
                        )
                    write_power_csv(ts.power_samples, cfg.server.power_agg_period_s, out_dir / "gpu_power.csv")
                    write_cdf_csv(ts.batch_sizes, out_dir / "cdf_batch.csv", kind="empirical")
                except Exception as e:
                    logging.info(f"Failed to write sampler outputs: {e}")
                logging.info(f"End trace: {trace} status={status} {error_note}")
                # Stop server after each trace (always)
                server_proc.terminate()
                try:
                    server_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logging.warning("Server did not exit in time; killing.")
                    server_proc.kill()

        logging.info("All bench configurations completed.")
    finally:
        # Ensure log file is flushed
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
