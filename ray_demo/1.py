# 并行任务处理
import ray
import time

# 如果已经启动了ray集群，就不需要再进行ray.init()
# 在使用ray相关的函数时，会自动连接到ray集群。

# 可以提交到ray集群里进行计算
@ray.remote
def remote_task(x):
    time.sleep(4)  # 模拟耗时操作
    return x * x

# 记录开始时间
start = time.time()

# 并行执行多个任务
result_ids = []
for i in range(4):
    result_ids.append(remote_task.remote(i))  # 异步调用，立即返回ID

# print(result_ids) [ObjectRef(d2b4e3e4a6f1ddc5ffffffffffffffffffffffff0700000001000000), 
# ObjectRef(11149dae5a34cd9affffffffffffffffffffffff0700000001000000),
# ObjectRef(a363a5a94a784cb3ffffffffffffffffffffffff0700000001000000),
# ObjectRef(b6fe777c23eaf580ffffffffffffffffffffffff0700000001000000)]

# 获取结果（会阻塞直到所有任务完成）
results = ray.get(result_ids) 
print(f"Results: {results}")

# 计算耗时
print(f"Time used: {time.time() - start:.2f}s")