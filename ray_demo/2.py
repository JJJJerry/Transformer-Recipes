# 有状态计算：使用 Actor

import ray

# 定义一个Actor类（保存状态的远程对象）
@ray.remote
class Counter:
    def __init__(self,start_value):
        self.value = start_value
        
    def increment(self):
        self.value += 1
        return self.value
    
    def get_value(self):
        return self.value

# 创建Actor实例
counter = Counter.remote(start_value=20)

# 并发调用方法（注意：Actor的方法调用是顺序执行的）
results = []
for _ in range(5):
    results.append(counter.increment.remote())

# 获取最终值
print("Final value:", ray.get(counter.get_value.remote()))  # 输出 5