# 分布式数据处理
import ray
import numpy as np

# 创建一个大数组并放入共享内存
data = np.random.rand(20000)  # 很多个数字
data_ref = ray.put(data)  # 将数据存入共享内存，避免多次传输

# print(ray.get(data_ref)[0]) # 可以通过ray.get拿到
# print(type(data_ref)) # <class 'ray._raylet.ObjectRef'>

@ray.remote
def process_chunk(data_ref, start, end):
    # 直接传递ref
    # 在remote函数里面
    # 不需要chunk = ray.get(data_ref)[start:end]
    chunk = data_ref[start:end]  # 获取数据片段
    return np.sum(chunk)  # 计算数据片段的求和

# 将数据分成4个块并行处理
chunk_size = len(data) // 4
futures = []
for i in range(4):
    start = i * chunk_size
    end = (i+1) * chunk_size if i < 3 else len(data)
    futures.append(process_chunk.remote(data_ref, start, end))

# 汇总结果
partial_sums = ray.get(futures)
total = sum(partial_sums)
print(f"Total sum: {total:.4f} (Ground truth: {np.sum(data):.4f})")
