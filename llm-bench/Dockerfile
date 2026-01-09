FROM vllm/vllm-openai:latest

# All work under /workspace
WORKDIR /workspace

# Prevent any remote downloads at runtime
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1

# Install a tiny HTTP client lib for the benchmark script
RUN pip install --no-cache-dir "httpx>=0.27"

# Copy model configs (downloaded locally beforehand)
COPY models/llama2-7b /opt/models/llama2-7b
COPY models/llama2-70b /opt/models/llama2-70b
COPY queries /workspace/queries

# Copy scripts and example queries
COPY server.py /workspace/server.py
COPY client.py /workspace/client.py
COPY run.py /workspace/run.py
COPY config.py /workspace/config.py
COPY run.sh /workspace/run.sh
RUN chmod +x /workspace/run.sh

# Default command: start vLLM server
ENTRYPOINT ["/workspace/run.sh"]
