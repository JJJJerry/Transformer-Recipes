# CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com torchrun --nnodes 1 --node_rank 0 --nproc_per_node 4 --master_addr 127.0.0.1 --master_port 10010 sft_demo_transformers.py
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset
import torch
import json

class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                messages = [
                    {"role": "user", "content": entry["instruction"]},
                    {"role": "assistant", "content": entry["output"]}
                ]
                
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except Exception as e:
                    print(f"Error formatting text: {e}")
                    continue

                tokenized = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_attention_mask=False
                )
                input_ids = tokenized["input_ids"][0]
                
                assistant_response = f"{messages[-1]['content']}<|im_end|>\n"
                assistant_token_len = len(tokenizer.encode(
                    assistant_response,
                    add_special_tokens=False
                ))
                
                labels = input_ids.clone()
                labels[:len(labels)-assistant_token_len] = -100
                
                self.data.append({
                    "input_ids": input_ids,
                    "labels": labels
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "labels": self.data[idx]["labels"]
        }

class DataCollatorForSFT:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch): # 让实例像函数的方式一样使用
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        padded_inputs = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding="longest",
            return_tensors="pt",
        )
        
        padded_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding="longest",
            return_tensors="pt",
            return_attention_mask=False
        )["input_ids"]
        
        padded_labels[padded_labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": padded_labels
        }

# 参数配置
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "./test_data/identity.json"
MAX_LENGTH = 2048
BATCH_SIZE = 4
LR = 1e-5
EPOCHS = 3

# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    pad_token="<|endoftext|>",
    padding_side="right"
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 准备数据
dataset = SFTDataset(DATA_PATH, tokenizer, MAX_LENGTH)
data_collator = DataCollatorForSFT(tokenizer)

# 训练配置
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    bf16=True,
    deepspeed='./deepspeed_config/deepspeed_z2.json',
    logging_dir="./logs",
    save_strategy="no",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    report_to="none"
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model("qwen-sft")