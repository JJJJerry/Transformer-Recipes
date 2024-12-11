import torch
from torch import nn
import numpy as np
import random
def get_dataset_add1(data_num=2000,vocab_size=16,seq_len=8,batch_first=False):
    """
        生成一个加法任务的数据集，输入是一个长度为seq_len的随机序列，输出是输入序列中每个数字加1的结果。
        比如:
            src: 1 2 3 4 5 6 7 8
            tgt: 2 3 4 5 6 7 8 9
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
        比如:
            src: 1 2 3 4 5 6 7 8
            tgt: 1 2 3 4 5 6 7 8
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

def get_dataset_AddSeq(data_num=2000):
    """
        生成一个加法任务数据集，输入是两个个位数的数字，输出是这两个数字的和。  
        
        index_add=10
        
        index_pad=11
        
        index_bos=12

        index_eos=13
        
        比如：  
            src: 12 4 10 1 13  
            tgt: 12 5 13  
            
            src: 12 6 10 7 13  
            tgt: 12 1 3 13  
    """
    dataset=[]
    index_add=10
    vocab_size=14
    index_pad=11
    index_bos=12
    index_eos=13
    def digital2list(digits:int):
        if digits==0:
            return [0]
        res=[]
        while digits>0:
            res.append(digits%10)
            digits//=10
        return res[::-1] # 把列表倒过来
    for _ in range(data_num):    
        a = random.randint(0, 9)
        b = random.randint(0, 9)
        c = a + b
        input_seq = torch.tensor([index_bos] + digital2list(a) + [ index_add ] +digital2list(b) + [index_eos])
        target_seq = torch.tensor([index_bos] + digital2list(c) + [ index_eos])
        dataset.append(
            {
                'src':input_seq,
                'tgt':target_seq
            }
        )
    def collate_fn(batch):
        src=[b['src'] for b in batch]
        tgt=[b['tgt'] for b in batch]
        src=nn.utils.rnn.pad_sequence(src,padding_value=index_pad,batch_first=True)
        tgt=nn.utils.rnn.pad_sequence(tgt,padding_value=index_pad,batch_first=True)
        
        batch={
                'src':src,
                'tgt':tgt
            }
        return batch
    return dataset,collate_fn,(index_bos,index_eos,index_pad,index_add,vocab_size)