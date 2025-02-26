from dataclasses import dataclass, field
from transformers import HfArgumentParser
import os
import sys
# 定义参数类
@dataclass
class MyArguments:
    learning_rate: float = field(metadata={"help": "The learning rate for training."})
    batch_size: int = field(metadata={"help": "Batch size for training."})
    num_epochs: int = field(default=3, metadata={"help": "Number of epochs to train for."})

# 创建 HfArgumentParser 实例
parser = HfArgumentParser(MyArguments)

# 解析命令行参数
args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))[0]

# 打印解析后的参数
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size: {args.batch_size}")
print(f"Number of Epochs: {args.num_epochs}")