# 机器学习场景
import ray
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 加载数据
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

@ray.remote
def train_model(n_estimators):
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return (n_estimators, score)

# 并行测试不同参数
params = [10, 20, 50, 100, 200]
futures = [train_model.remote(p) for p in params]
results = ray.get(futures)

# 输出最佳参数
best = max(results, key=lambda x: x[1])
print(f"Best params: n_estimators={best[0]}, accuracy={best[1]:.4f}")
