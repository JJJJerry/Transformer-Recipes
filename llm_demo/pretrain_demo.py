from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from utils import load_jsonl
from torch.utils.data import Dataset
from tqdm import tqdm
model_name = "Qwen/Qwen2.5-0.5B"
device = "cuda:5"

config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name)
if config.pad_token_id is None:
    config.pad_token_id = (
        config.eos_token_id
    )  # 避免提示：Setting pad_token_id to eos_token_id:None for open-end generation.
model = AutoModelForCausalLM.from_config(config=config).to(device)  # 用from_config方法，参数都是初始化的
#model = AutoModelForCausalLM.from_pretrained(model_name,device_map=device) # 参数
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = load_jsonl(path="../data/webText2019zh_1k.jsonl")
train_data_num=int(len(data)*0.9)
train_texts = data[:train_data_num]
eval_texts = data[train_data_num:]

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]["text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        # 获取 input_ids 和 attention_mask
        input_ids = encoding["input_ids"].squeeze()  # 去掉批量维度
        attention_mask = encoding["attention_mask"].squeeze()

        return {"input_ids": input_ids, "attention_mask": attention_mask}


train_dataset = TextDataset(train_texts, tokenizer, max_length=512)
eval_dataset = TextDataset(eval_texts, tokenizer, max_length=512)

batch_size = 4
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size
)
eval_dataloader = torch.utils.data.DataLoader(
    dataset=eval_dataset, batch_size=batch_size
)


def train(model, train_dataloader, loss_func, optimizer, device):
    model.train()
    losses = []
    progress_bar = tqdm(train_dataloader, desc="Training", total=len(train_dataloader))
    for step,batch in enumerate(progress_bar):
        inputs_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        text = inputs_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_input = text[:, :-1]  # 模型的输入是text的前n-1个token
        logits = model.forward(
            input_ids=text_input, attention_mask=attention_mask
        ).logits
        text_expected = text[:, 1:]  # 模型的期望输出是text的第2个token到第n个token
        text_expected = text_expected.reshape(-1)
        logits = logits.view(-1, logits.shape[-1])
        optimizer.zero_grad()
        loss = loss_func(logits, text_expected)
        loss.backward()
        optimizer.step()
        if step%100==0:
            generate(model,tokenizer)
        losses.append(loss.item())
        progress_bar.set_postfix(loss=loss.item()/batch_size, refresh=True)
    return sum(losses) / (len(losses) * batch_size)

@torch.inference_mode()
def generate(model:AutoModelForCausalLM,tokenizer:AutoTokenizer,sentence='我今天心情很'):
    input_ids=tokenizer(sentence,return_tensors="pt").to(device)
    model_output_ids=model.generate(
        **input_ids,
        max_new_tokens=128
    ) # (batch_size,seq_len)
    model_output_sentence=tokenizer.batch_decode(model_output_ids)[0]
    print(model_output_sentence)

@torch.inference_mode()  # 验证的时候关闭梯度计算
def eval(model, eval_dataloader, loss_func, device):
    model.eval()
    losses = []
    progress_bar = tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader))
    for step,batch in enumerate(progress_bar):
        inputs_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        attention_mask = attention_mask.to(device)
        text = inputs_ids.to(device)
        text_input = text[:, :-1]  # 模型的输入是text的前n-1个token
        logits = model.forward(
            input_ids=text_input, attention_mask=attention_mask
        ).logits
        text_expected = text[:, 1:]  # 模型的期望输出是text的第2个token到第n个token
        text_expected = text_expected.reshape(-1)
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_func(logits, text_expected)
        progress_bar.set_postfix(loss=loss.item()/batch_size, refresh=True)
        losses.append(loss.item())
    return sum(losses) / (len(losses) * batch_size)

loss_func = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-6)

epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train(
        model=model,
        train_dataloader=train_dataloader,
        loss_func=loss_func,
        optimizer=optimizer,
        device=device,
    )
    eval_loss = eval(
        model=model, eval_dataloader=eval_dataloader, loss_func=loss_func, device=device
    )
    print(f"Epoch:{epoch + 1}, train_loss:{train_loss}, eval_loss:{eval_loss}")

# 保存模型和tokenizer到指定目录
save_directory = f'qwen-2.5-0.5b-pt'
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)