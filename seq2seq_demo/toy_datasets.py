import torch
from torch import nn
import numpy as np
import random
def get_dataset_add1(data_num=2000,vocab_size=16,seq_len=8,batch_first=False):
    """
        生成一个加法任务的数据集，输入是一个长度为seq_len的随机序列，输出是输入序列中每个数字加1的结果。
    """
    dataset=[]
    for _ in range(data_num):
        src=torch.tensor([random.randint(0,vocab_size-2) for i in range(seq_len)])
        tgt=src+1
        dataset.append(
            {
                'src':src,
                'tgt':tgt
            }
        )
    def collate_fn(batch):
        src=[b['src'] for b in batch]
        tgt=[b['tgt'] for b in batch]
        src=nn.utils.rnn.pad_sequence(src,batch_first=batch_first)
        tgt=nn.utils.rnn.pad_sequence(tgt,batch_first=batch_first)
        batch={
                'src':src,
                'tgt':tgt
            }
        return batch
    return dataset,collate_fn

def get_dataset_repeat(data_num=2000,vocab_size=16,seq_len=8,batch_first=False):
    """
        生成一个重复数字的数据集，输入是一个长度为seq_len的随机序列，输出序列和输入序列一样的结果。
    """
    dataset=[]
    for _ in range(data_num):
        src=torch.tensor([random.randint(0,vocab_size-2) for i in range(seq_len)])
        tgt=src
        dataset.append(
            {
                'src':src,
                'tgt':tgt
            }
        )
    def collate_fn(batch):
        src=[b['src'] for b in batch]
        tgt=[b['tgt'] for b in batch]
        src=nn.utils.rnn.pad_sequence(src,batch_first=batch_first)
        tgt=nn.utils.rnn.pad_sequence(tgt,batch_first=batch_first)
        batch={
                'src':src,
                'tgt':tgt
            }
        return batch
    return dataset,collate_fn
