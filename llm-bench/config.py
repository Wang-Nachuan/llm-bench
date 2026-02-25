from dataclasses import dataclass, field
import os


@dataclass
class ServerConfig:
    # Core benchmark knobs (configured in this file; not via env vars)
    model_size: str = "7b"  # "7b" or "70b"
    port: int = 8000
    host: str = "0.0.0.0"
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    gpu_type: str = "a100"   # "a100" or "h100"
    max_model_len: int = 4096

    # Predefined values
    max_num_batched_tokens: int = 256       # Maximun tokens for chunked prefill
    max_num_seqs: int = 64                  # Maximun tokens for continuous batching
    gpu_memory_utilization: float = 0.90
    
    power_sample_period_s: float = 1.0
    batch_sample_period_s: float = 1.0
    power_agg_period_s: float = 10.0

    @property
    def model_dir(self) -> str:
        root_dir = os.getenv("ROOT_DIR", "/workspace")
        return f"{root_dir}/models/llama2-{self.model_size}"

    @property
    def served_model_name(self) -> str:
        return f"llama2-{self.model_size}"


@dataclass
class ClientConfig:
    queries_path: str  # can be a single CSV file or a directory with multiple CSVs
    base_url: str
    model_name: str
    results_root: str = "/workspace/bench_out"


@dataclass
class OrchestratorConfig:
    server: ServerConfig
    client: ClientConfig


@dataclass()
class BenchRunConfig:
    """One benchmark configuration to run (outer loop in run.py)."""
    name: str
    trace_folder: str  # e.g., "dev_8", "dev_16", "test"
    server_overrides: dict[str, object] = field(default_factory=dict)
    extra_server_args: list[str] = field(default_factory=list)


def bench_run_configs() -> list[BenchRunConfig]:
    # Keep this list small and explicit; add more entries as needed.
    return [
        # Test on 1 GPU
        # BenchRunConfig(
        #     name="test_dev_1_w16a16",
        #     trace_folder="test_dev_1",
        #     server_overrides={
        #         "model_size": "7b",
        #         "tp_size": 1,
        #         "pp_size": 1,
        #         "dp_size": 1,
        #     },
        #     extra_server_args=[],
        # ),
        # BenchRunConfig(
        #     name="test_dev_1_w8a16",
        #     trace_folder="test_dev_1",
        #     server_overrides={
        #         "model_size": "7b",
        #         "tp_size": 1,
        #         "pp_size": 1,
        #         "dp_size": 1,
        #     },
        #     extra_server_args=["--quantization", "fp8"],
        # ),

        # Test on 4 GPU
        # BenchRunConfig(
        #     name="test_dev_4_w16a16",
        #     trace_folder="test_dev_4",
        #     server_overrides={
        #         "model_size": "70b",
        #         "tp_size": 1,
        #         "pp_size": 4,
        #         "dp_size": 1,
        #     },
        #     extra_server_args=[],
        # ),
        # BenchRunConfig(
        #     name="test_dev_4_w8a16",
        #     trace_folder="test_dev_4",
        #     server_overrides={
        #         "model_size": "70b",
        #         "tp_size": 1,
        #         "pp_size": 4,
        #         "dp_size": 1,
        #     },
        #     extra_server_args=["--quantization", "fp8"],
        # ),

        # Test on 8 GPUs
        # BenchRunConfig(
        #     name="test_dev_8_w16a16",
        #     trace_folder="test_dev_8",
        #     server_overrides={
        #         "model_size": "70b",
        #         "tp_size": 1,
        #         "pp_size": 4,
        #         "dp_size": 2,
        #     },
        #     extra_server_args=[],
        # ),
        # BenchRunConfig(
        #     name="test_dev_8_w8a16",
        #     trace_folder="test_dev_8",
        #     server_overrides={
        #         "model_size": "70b",
        #         "tp_size": 1,
        #         "pp_size": 4,
        #         "dp_size": 2,
        #     },
        #     extra_server_args=["--quantization", "fp8"],
        # ),
        
        # Full benchmark on 8 GPUs
        BenchRunConfig(
            name="dev_8_w16a16",
            trace_folder="dev_8",
            server_overrides={
                "model_size": "70b",
                "tp_size": 1,
                "pp_size": 4,
                "dp_size": 2,
            },
            extra_server_args=[],
        ),
        BenchRunConfig(
            name="dev_8_w8a16",
            trace_folder="dev_8",
            server_overrides={
                "model_size": "70b",
                "tp_size": 1,
                "pp_size": 4,
                "dp_size": 2,
            },
            extra_server_args=["--quantization", "fp8"],
        ),
    ]


def default_orchestrator_config() -> OrchestratorConfig:
    # Defaults optimized for the local single-GPU test path
    port = int(os.getenv("VLLM_PORT", "8000"))
    host = os.getenv("VLLM_HOST", "0.0.0.0")
    max_model_len = int(os.getenv("MAX_MODEL_LEN", "4096"))

    base_url = f"http://127.0.0.1:{port}/v1"
    model_name = "llama2-7b"
    # Default to a directory of traces
    queries_path = os.getenv("QUERIES_PATH", "/workspace/queries")
    results_root = os.getenv("RESULTS_ROOT", "/workspace/bench_out")

    server_cfg = ServerConfig(
        port=port,
        host=host,
        max_model_len=max_model_len,
    )
    client_cfg = ClientConfig(
        queries_path=queries_path,
        base_url=base_url,
        model_name=model_name,
        results_root=results_root,
    )
    return OrchestratorConfig(server=server_cfg, client=client_cfg)


