export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/tf/llm-test/data
python hf_data.py --count 10 --seed 42 --output swe_prompts.jsonl