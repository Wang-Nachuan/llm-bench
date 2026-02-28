sudo rm -rf bench_out
mkdir bench_out

# No runtime env parameters: configuration lives in llm-bench/run.sh.
# For local convenience, mount the fixed in-container output dir to ./bench_out.
sudo docker run --rm --gpus all --ipc=host \
    -v "$(pwd)/bench_out:/workspace/bench_out" \
    llm-bench:latest