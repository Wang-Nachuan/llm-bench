from dataclasses import dataclass, field
import os
from typing import List


@dataclass
class ServerConfig:
    model_size: str = "70b"  # "7b" or "70b"
    port: int = 8000
    host: str = "0.0.0.0"
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    gpu_type: str = "a100"   # "a100" or "h100"
    max_model_len: int = 4096

    # Predefined values
    max_num_batched_tokens: int = 16384     # Maximun tokens for chunked prefill
    max_num_seqs: int = 128                 # Maximun tokens for continuous batching
    gpu_memory_utilization: float = 0.93

    @property
    def model_dir(self) -> str:
        return f"/opt/models/llama2-{self.model_size}"

    @property
    def served_model_name(self) -> str:
        return f"llama2-{self.model_size}"


@dataclass
class ClientConfig:
    queries_path: str  # can be a single CSV file or a directory with multiple CSVs
    base_url: str
    model_name: str
    results_root: str = "/workspace/bench_out"
    power_sample_period_s: float = 1.0
    power_agg_period_s: float = 10.0


@dataclass
class OrchestratorConfig:
    server: ServerConfig
    client: ClientConfig


def default_orchestrator_config() -> OrchestratorConfig:
    # Defaults optimized for the local single-GPU test path
    port = int(os.getenv("VLLM_PORT", "8000"))
    host = os.getenv("VLLM_HOST", "0.0.0.0")
    model_size = os.getenv("MODEL_SIZE", "7b")  # orchestrated default: 7B
    tp_size = int(os.getenv("TP_SIZE", "1"))
    pp_size = int(os.getenv("PP_SIZE", "1"))
    dp_size = int(os.getenv("DP_SIZE", "1"))
    gpu_type = os.getenv("GPU_TYPE", "a100").lower()
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))

    base_url = f"http://127.0.0.1:{port}/v1"
    model_name = f"llama2-{model_size}"
    # Default to a directory of traces
    queries_path = os.getenv("QUERIES_PATH", "/workspace/queries")
    results_root = os.getenv("RESULTS_ROOT", "/workspace/bench_out")
    power_sample_period_s = float(os.getenv("POWER_SAMPLE_PERIOD_S", "1.0"))
    power_agg_period_s = float(os.getenv("POWER_AGG_PERIOD_S", "10.0"))

    server_cfg = ServerConfig(
        model_size=model_size,
        port=port,
        host=host,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        gpu_type=gpu_type,
        max_model_len=max_model_len,
    )
    client_cfg = ClientConfig(
        queries_path=queries_path,
        base_url=base_url,
        model_name=model_name,
        results_root=results_root,
        power_sample_period_s=power_sample_period_s,
        power_agg_period_s=power_agg_period_s,
    )
    return OrchestratorConfig(server=server_cfg, client=client_cfg)


