# server.py
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

from config import ServerConfig

def build_server_cmd(cfg: ServerConfig, *, extra_args: List[str] | None = None) -> List[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model_dir,
        "--served-model-name", cfg.served_model_name,
        "--dtype", "float16",
        "--kv-cache-dtype", "auto",
        "--load-format", "dummy",          # random FP16 weights, no stored checkpoints
        "--max-model-len", str(cfg.max_model_len),
        "--tensor-parallel-size", str(cfg.tp_size),
        "--pipeline-parallel-size", str(cfg.pp_size),
        "--data-parallel-size", str(cfg.dp_size),
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--disable-log-requests",          # reduce logging overhead
        "--no-enable-prefix-caching",
        "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return cmd

def launch_server(cfg: ServerConfig, *, extra_args: List[str] | None = None) -> subprocess.Popen:
    cmd = build_server_cmd(cfg, extra_args=extra_args)
    print("Launching vLLM server with command:")
    print(" ".join(cmd))
    log_path = Path(os.getenv("VLLM_LOG_FILE", "/workspace/bench_out/vllm.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", encoding="utf-8", buffering=1)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _tee(src, dst):
        for line in src:
            log_f.write(line)
            dst.write(line)
            dst.flush()

    threading.Thread(target=_tee, args=(proc.stdout, sys.stdout), daemon=True).start()
    threading.Thread(target=_tee, args=(proc.stderr, sys.stderr), daemon=True).start()
    threading.Thread(target=lambda: (proc.wait(), log_f.close()), daemon=True).start()
    return proc
