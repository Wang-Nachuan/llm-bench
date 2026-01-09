export HF_HOME=/home/nachuan3/hf_cache
export TOKEN=

hf auth login --token $TOKEN
echo "User:"
hf auth whoami