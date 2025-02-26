import torch
from torch import nn
import math
# 多头注意力
index_pad=0
# 多头注意力
class MultiHeadAttention(nn.Module):
    """
    多头注意力层，用于计算多个注意力头的输出
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 除取整

        self.W_q = nn.Linear(d_model, d_model)  # （输入的特征维度，输出的特征维度）
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算多头注意力的输出
        """
        # Q (N,n_head,S,d_k)
        # K (N,n_head,S,d_k)
        # V (N,n_head,S,d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # attn_scores (N,n_head,S,S)
        if mask is not None: 
            # mask (N,1,1,S)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # attn_scores里mask为0的地方，用负无穷填充
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # attn_probs (N,n_head,S,S)
        output = torch.matmul(attn_probs, V)
        # output (N,n_head,S,d_k)
        return output

    def split_heads(self, x):
        """
        分割多头注意力的输入，将输入的特征维度分割成多个头
        """
        # (N,S,D)
        batch_size, seq_length, d_model = x.size()
        # (N,S,n_head,d_k)
        # (N,n_head,S,d_k)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        # 若将tensor的第一维与第二维转置，则：
        # x_new[i][j][k] = x[i][k][j]
        # x_new[i][0][k] = x[i][k][0]
        # x_new[i][1][k] = x[i][k][1]
        # x_new[i][2][k] = x[i][k][2]
        # x_new[i][3][k] = x[i][k][3]

    def combine_heads(self, x):
        """
        组合多头注意力的输出，将多个头的输出拼接起来
        """
        # x (N,n_head,S,d_k)
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Q (N,S,D)
        Q = self.split_heads(self.W_q(Q))
        # Q (N,n_head,S,d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # attn_output (N,n_head,S,d_k)
        output = self.W_o(self.combine_heads(attn_output))

        return output


class PositionWiseFeedForward(nn.Module):
    """
    前馈神经网络，包含两个全连接层和ReLU激活函数
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))) 


# 位置编码
class PositionalEncoding(nn.Module):
    """
    位置编码，用于将输入序列中的每个位置映射到一个向量中。解决了原版attention中学习不到位置信息的问题。
    """
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(
            max_seq_length, d_model
        )  # 最大序列长度为行，特征维度为列，一个矩阵，和输入一样
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 所有行取偶数索引（奇数列）
        pe[:, 1::2] = torch.cos(position * div_term)  # 所有行取奇数索引

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size) # 分类头
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, tgt)->torch.Tensor:
        # tgt (N,T)
        tgt_mask = (tgt != index_pad).unsqueeze(1).unsqueeze(3) # (N,1,T,1)
        seq_length = tgt.size(1)
        nopeak_mask = (
            1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)
        ).bool().to(tgt.device)
        # decoder的遮蔽未来信息
        tgt_mask = tgt_mask & nopeak_mask
    
        return tgt_mask

    def forward(self, tgt)->torch.Tensor:
        tgt_mask = self.generate_mask(tgt) # 根据src和tgt，生成mask
        tgt_embedded = self.dropout(self.positional_encoding(self.embedding(tgt)))

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output,tgt_mask) # decoder这里做的是cross attention，所以需要src_mask
        output = self.fc_out(dec_output)
        return output
    @torch.inference_mode()
    def predict(self,tgt:torch.Tensor,max_seq_length:int,index_eos:int)->list:
        batch_size,tgt_len=tgt.shape
        output=[]
        for i in range(tgt_len,max_seq_length):
            logits=self.forward(tgt) # (batch_size,tgt_len,vocab_size)
            next_token=logits[:,-1,:].argmax(dim=-1) 
            output.append(next_token.item())
            if next_token.item()==index_eos:
                break
            tgt=torch.cat([tgt,next_token.unsqueeze(0)],dim=-1)
        return output