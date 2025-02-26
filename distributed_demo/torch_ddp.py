import torch
import sys
import os

sys.path.append(os.getcwd())
from seq2seq_demo.toy_datasets import get_dataset_AddSeq4DecoderOnly
from transformer import Transformer
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import time

@torch.inference_mode()
def add(a, b, model):
    model.eval()
    input_list = [index_bos] + [a, index_add, b] + [index_equal]
    text = torch.tensor(input_list).to(device).unsqueeze(0)  # (1,seq_len)
    output = model.predict(text, max_seq_length=max_seq_length, index_eos=index_eos)
    print(f"输入序列 {input_list}")
    print(f"输出序列 {input_list+output}")
    print(f"{a} + {b}={output[:-1]}")


def train(model, train_dataloader, loss_func, optimizer, device):
    model.train()
    losses = []
    for batch in train_dataloader:
        text = batch["text"].to(device)
        #print(text.shape)
        #sys.exit()
        text_input = text[:, :-1]  # 模型的输入是text的前n-1个token
        logits = model(text_input)
        text_expected = text[:, 1:]  # 模型的期望输出是text的第2个token到第n个token
        text_expected = text_expected.reshape(-1)
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_func(logits, text_expected)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return sum(losses) / (len(losses) * batch_size)


@torch.inference_mode()  # 验证的时候关闭梯度计算
def eval(model, eval_dataloader, loss_func, device):
    model.eval()
    losses = []
    for batch in eval_dataloader:
        text = batch["text"].to(device)
        text_input = text[:, :-1]  # 模型的输入是text的前n-1个token
        logits = model(text_input)
        text_expected = text[:, 1:]  # 模型的期望输出是text的第2个token到第n个token
        text_expected = text_expected.reshape(-1)
        logits = logits.view(-1, logits.shape[-1])
        loss = loss_func(logits, text_expected)
        losses.append(loss.item())
    return sum(losses) / (len(losses) * batch_size)


def init_distributed_mode():
    dist.init_process_group(backend="nccl")
    device = f"cuda:{dist.get_rank()}"
    torch.cuda.set_device(device)
    return device


if __name__ == "__main__":
    data_num = 10000
    (
        dataset,
        collate_fn,
        (index_bos, index_eos, index_pad, index_add, index_equal, vocab_size),
    ) = get_dataset_AddSeq4DecoderOnly(data_num=data_num)

    train_size = 0.2
    train_dataset = dataset[: int(data_num * train_size)]
    eval_dataset = dataset[int(data_num * train_size) :]

    device = init_distributed_mode()
    batch_size = 64
    train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )
    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )
    torch.manual_seed(42)  # 设置随机种子以确保可重复性
    # 设置所有GPU的随机种子
    torch.cuda.manual_seed_all(42)
    num_heads = 8
    d_ff = 128
    d_model = 64
    num_layers = 3
    dropout = 0.1
    max_seq_length = 16 
    
    transformer = Transformer(
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ).to(device)
    model = DistributedDataParallel(transformer)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=index_pad)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9
    )

    epochs = 10  
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        now=time.time()
        train_loss = train(
            model=model,
            train_dataloader=train_dataloader,
            loss_func=loss_func,
            optimizer=optimizer,
            device=device,
        )
        a = 6
        b = 8
        if dist.get_rank() == 0:
            eval_loss = eval(
            model=transformer,
            eval_dataloader=eval_dataloader,
            loss_func=loss_func,
            device=device)
            print(f"epoch:{epoch}, train_loss:{train_loss}, eval_loss:{eval_loss}")
            add(a, b, transformer)
            print(f'用时 {time.time()-now}s')
