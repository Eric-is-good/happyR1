# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
from datasets import Dataset
import json
import re

##########################################################################################################
# process data
FORMART_PROMPT = """
请按照这样的格式回答问题:
<think>
详细思考过程
</think>
<answer>
简短的先总结，再回答，最终答案是一个数字，使用'#### 数字'表示 
</answer>
"""

file_path = 'data.jsonl'  # Replace with the path to your uploaded JSON file

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f.readlines()]
    return data

data = load_jsonl(file_path)

data_dict = {
    ('prompt' if key == 'question' else key): 
    [
        [{"role": "user", "content": FORMART_PROMPT + item[key]}] if key == 'question' else
        [{"role": "assistant", "content": item[key]}] for item in data
    ]
    for key in data[0].keys()
}

dataset = Dataset.from_dict(data_dict)

print(dataset)
print(dataset[0])

# ##########################################################################################################
# # reward model/fuctions ( only fuctions @_@ )

def length_reward_func(completions, **kwargs):
    score = [float(len(completion[0]["content"]))*0.005 for completion in completions]
    
    # print("##############################")
    # print(score)
    
    return score

def format_reward_func(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    score = [1.0 if match else 0.0 for match in matches]
    
    # print("##############################")
    # print(score)
    
    return score

def answer_reward_func(completions, answer, **kwargs):
    completions_matches = [re.search(r"####\s*(\d+)", str(c)) for c in completions]
    completions_ = [match.group(1) if match else "" for match in completions_matches]
    
    ground_truth_matches = [re.search(r"####\s*(\d+)", str(c)) for c in answer]
    ground_truth_ = [match.group(1) if match else "" for match in ground_truth_matches]

    score = [3.0 if c.strip() == gt.strip() else 0.0 for c, gt in zip(completions_, ground_truth_)]
    
    # print("##############################")
    # print(score)
    
    return score


##########################################################################################################
# training args

training_args = GRPOConfig(
    output_dir="Qwen2.5-3B-GRPO", 
    run_name="Qwen2.5-3B-GRPO",
    bf16=True,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    logging_steps=10
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    reward_funcs=[length_reward_func, format_reward_func, answer_reward_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
