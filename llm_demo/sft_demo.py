from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch
import json
from tqdm import tqdm

class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data=json.load(f)
            for entry in data:
                # 构造对话格式
                messages = [
                    {"role": "user", "content": entry["instruction"]},
                    {"role": "assistant", "content": entry["output"]}
                ]
                
                # 应用Qwen的chat_template
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except Exception as e:
                    print(f"Error formatting text: {e}")
                    continue
                
                # Tokenize完整文本
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
                
                # 生成labels
                labels = input_ids.clone()
                # 将prompt部分（包括用户内容和分隔符）设为-100
                labels[:len(labels)-assistant_token_len] = -100
                # 确保padding部分也为-100（虽然当前还未padding）
                
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

def collate_fn(batch, tokenizer):
    # 右侧padding
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding="longest",
        return_tensors="pt",
        # pad_to_multiple_of=8 # 将序列长度填充到该值的整数倍（如8的倍数，用于优化GPU计算）。
    ) # 返回input_ids和attention_mask
    
    padded_labels = tokenizer.pad(
        {"input_ids": labels},
        padding="longest",
        return_tensors="pt",
        # pad_to_multiple_of=8,
        return_attention_mask=False
    )["input_ids"]
    
    # 将padding部分的labels设为-100
    padded_labels[padded_labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "labels": padded_labels
    }

# 参数配置
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # 按需选择模型大小
DATA_PATH = "llm_demo/test_data/identity.json"
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
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 准备数据
dataset = SFTDataset(DATA_PATH, tokenizer, MAX_LENGTH)
data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    collate_fn=lambda batch: collate_fn(batch, tokenizer),
    shuffle=True
)

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 训练循环
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        inputs = batch["input_ids"].cuda()
        masks = batch["attention_mask"].cuda()
        labels = batch["labels"].cuda()
        
        outputs = model(
            input_ids=inputs,
            attention_mask=masks,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

# 保存模型
model.save_pretrained("qwen-sft")
tokenizer.save_pretrained("qwen-sft")