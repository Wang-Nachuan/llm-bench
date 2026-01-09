# server.py
import subprocess
import sys
from typing import List

from config import ServerConfig

def build_server_cmd(cfg: ServerConfig) -> List[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", cfg.model_dir,
        "--served-model-name", cfg.served_model_name,
        "--dtype", "float16",
        "--load-format", "dummy",          # random FP16 weights, no stored checkpoints
        "--max-model-len", str(cfg.max_model_len),
        "--tensor-parallel-size", str(cfg.tp_size),
        "--pipeline-parallel-size", str(cfg.pp_size),
        "--data-parallel-size", str(cfg.dp_size),
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--disable-log-requests",          # reduce logging overhead
        "--no-enable-prefix-caching",
    ]
    return cmd

def launch_server(cfg: ServerConfig) -> subprocess.Popen:
    cmd = build_server_cmd(cfg)
    print("Launching vLLM server with command:")
    print(" ".join(cmd))
    return subprocess.Popen(cmd)
