export HF_HOME=/home/nachuan3/hf_cache

rm -rf llm-bench/models
mkdir -p llm-bench/models/llama2-7b llm-bench/models/llama2-70b

# 7B
hf download meta-llama/Llama-2-7b-hf \
  config.json tokenizer.model tokenizer_config.json special_tokens_map.json \
  --local-dir llm-bench/models/llama2-7b

# 70B
hf download meta-llama/Llama-2-70b-hf \
  config.json tokenizer.model tokenizer_config.json special_tokens_map.json \
  --local-dir llm-bench/models/llama2-70b

rm -rf llm-bench/models/llama2-7b/.cache
rm -rf llm-bench/models/llama2-70b/.cache