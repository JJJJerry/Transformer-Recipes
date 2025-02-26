from transformers import BertTokenizer, BertForSequenceClassification
import torch
device='cuda'
# 加载预训练的bert-base-chinese模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2).to(device)  # 假设是二分类情感分析任务

# 示例数据
texts = [
    "今天的天气真好，心情也特别好！",
    "这本书非常精彩，读完让我感触颇深！",
    "我终于完成了项目，感觉特别有成就感！",
    "和朋友们一起出去玩，真是太开心了！",
    "昨晚的演唱会超棒，歌手真是太有才了！",
    "我刚买了一部新手机，功能超强，真是太喜欢了！",
    "今天吃了特别好吃的火锅，太满足了！",
    "我和家人一起度过了一个美好的周末，非常温馨！",
    "今天工作顺利，老板夸奖了我，心情特别好！",
    "这部电影太感人了，结局让人热泪盈眶！",
    "今天下雨了，心情很低落，什么都不想做。",
    "这部电影真是浪费时间，剧情毫无亮点。",
    "我的手机坏了，真是倒霉，心情差到极点。",
    "工作压力太大，今天完全做不完任务，真的很沮丧。",
    "今天和朋友吵架了，心里很难受。",
    "这本书太无聊了，完全没看下去的兴趣。",
    "下班后累得不行，回家只想躺着，不想说话。",
    "昨天的演出太失望了，音乐和灯光都不好，完全不值票价。",
    "我刚刚被老板批评了，心情极差，完全没有动力做事。",
    "今天下班很晚，饿得不行，结果晚餐又没做成，真是糟糕的一天。"
]
# 1表示积极的情感，0表示消极的情感
labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
classes={1:"积极",0:"消极"}

from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

train_ids = tokenizer(train_texts, padding=True, truncation=True, return_tensors='pt')
# 这里是整个数据集一起tokenize。所有句子会被填充padding到一个数据集中最大的序列长度
# 如果数据集较大的话，一个batch一个batch的tokenize会比较好
test_ids = tokenizer(test_texts, padding=True, truncation=True, return_tensors='pt')
train_labels=torch.tensor(train_labels)
test_labels=torch.tensor(test_labels)


from datasets import Dataset
train_dataset=Dataset.from_dict({
    "input_ids":train_ids['input_ids'],
    "attention_mask":train_ids['attention_mask'],
    "label":train_labels
})
test_dataset=Dataset.from_dict({
    "input_ids":test_ids['input_ids'],
    "attention_mask":test_ids['attention_mask'],
    "label":test_labels
})

# 从 transformers 库中导入 TrainingArguments 类，用于设置训练参数
from transformers import TrainingArguments

# 设置训练参数
batch_size = 2 # 批大小
output_dir = "bert-base-chinese-finetuned-emotion" # 输出目录

# 创建 TrainingArguments 对象，设置训练参数
training_args = TrainingArguments(output_dir = output_dir, # 输出目录
                                 overwrite_output_dir=True, # 如果输出目录存在，则覆盖
                                 num_train_epochs=5, # 训练轮数
                                 learning_rate = 2e-5, # 学习率
                                 per_device_train_batch_size= batch_size, # 每个设备上的训练批大小
                                 per_device_eval_batch_size = batch_size, # 每个设备上的评估批大小
                                 logging_steps=1, # 日志记录步数
                                 save_strategy="no", # 不保存权重
                                 weight_decay=0.01, # 权重衰减
                                 eval_strategy = 'epoch', # 评估策略为每个 epoch 结束后评估
                                 report_to="none",
                                 disable_tqdm=False,) # 不禁用 tqdm 进度条

# 从 transformers 库中导入 Trainer 类，用于模型训练
from transformers import Trainer

# 创建 Trainer 对象，设置模型、训练参数、训练集、评估集和 tokenizer
trainer = Trainer(model=model, 
                  args=training_args, # 模型和训练参数
                  train_dataset=train_dataset, # 训练集
                  eval_dataset=test_dataset # 评估集
                  )
trainer.train()

@torch.inference_mode()
def predict(model,tokenizer,sentence):
    input_ids = tokenizer.encode_plus(sentence,return_tensors='pt').to(device)
    #print(model.bert(**input_ids)[1].shape)
    output=model(**input_ids)
    logits=output.logits
    print(logits)
    return classes[torch.argmax(logits).item()]
print(predict(model,tokenizer,sentence='心情很差'))