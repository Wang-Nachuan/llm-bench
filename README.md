# LLM Serving Benchmark Deployment

1. Pull the docker image:
    ```bash
    sudo docker pull nachuanwang/llm-bench:latest
    ```

2. Run on A100 cluster (machine type: [a2-megagpu-16g](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#a2-standard-vms)):
    ```bash
    sudo docker run --rm -it --gpus all --ipc=host \
        -e GPU_TYPE=a100 -e MODEL_SIZE=70b -e TP_SIZE=1 -e PP_SIZE=4 -e DP_SIZE=4 \
        -v "$(pwd)/bench_out:/mnt/out" \
        --entrypoint /bin/bash \
        nachuanwang/llm-bench:latest \
        /workspace/run.sh
    ```

3. Run on H100 cluster (machine type: [a3-edgegpu-8g](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines#a3-edge-vms), ideally with 400 Gbps networking):
    ```bash
    sudo docker run --rm -it --gpus all --ipc=host \
        -e GPU_TYPE=h100 -e MODEL_SIZE=70b -e TP_SIZE=1 -e PP_SIZE=4 -e DP_SIZE=2 \
        -v "$(pwd)/bench_out:/mnt/out" \
        --entrypoint /bin/bash \
        nachuanwang/llm-bench:latest \
        /workspace/run.sh
    ```


4. The benchmark will start automatically and finishes in 3.5-4 hours. It may report some out-of-memory errors but it's fine. When complete, all results are saved to the ```bench_out/``` folder created in the location where the Docker command was executed. Please share the ```bench_out/``` folder with me, thanks!