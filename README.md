# LLM Serving Benchmark Deployment

## Important Files
- `llm-bench/`: The source code.
- `llm-bench/run.sh`: The script that container automatically executes when it is launched.
- `llm-bench/run.py`: The main script that launches the benchmark.
- `build_docker.sh`: Build the docker image and copy all source code in `llm-bench/` into it (need to be run whenever `llm-bench/` is updated).
- `run_docker.sh`: Run the docker container.

## Build and Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Wang-Nachuan/llm-bench.git llm-bench
    cd llm-bench
    ```

2. Uncomment the desired configuration in `llm-bench/run.sh` for A100/H100.

3. Build the docker image:
    ```bash
    ./build_docker.sh
    ```

3. Run the container:
    ```bash
    ./run_docker.sh
    ```

4. The benchmark will start automatically and finishes in 3.5-4 hours. It may somtimes report out-of-memory errors but it's fine. When complete, all results are saved to the `/workspace/bench_out` folder inside container (I currently map this folder to a external path `$(pwd)/bench_out` in `run_docker.sh` for convenience). Please share the `bench_out/` folder with me, thanks!