import json
from tqdm import tqdm
import datasets

all_dataset = datasets.load_from_disk("data/fingpt-sentiment-train")["train"]

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

data_list = []
for item in all_dataset.to_pandas().itertuples():
    tmp = {}
    tmp["instruction"] = item.instruction
    tmp["input"] = item.input
    tmp["output"] = item.output
    data_list.append(tmp)

with open("data/dataset_new.jsonl", 'w') as f:
    for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')

import json

import datasets
from transformers import AutoTokenizer, AutoConfig

model_name = "base_models/Llama-2-13b-hf"
jsonl_path = "data/dataset_new.jsonl"
save_path = 'data/dataset_new'
max_seq_length = 512
skip_overlength = True

def preprocess(tokenizer, config, example, max_seq_length):
    prompt = example["context"]
    
    target = example["target"]
    
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False
    )
        
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

def read_jsonl(path, max_seq_length, skip_overlength=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue    
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature



dataset = datasets.Dataset.from_generator(
    lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
)
dataset.save_to_disk(save_path)


