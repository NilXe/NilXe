from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # 隐藏层神经元数量组合
    'activation': ['relu', 'tanh'],  # 激活函数
    'solver': ['sgd', 'adam'],  # 优化器
    'alpha': [0.0001, 0.001, 0.01]  # 正则化参数
}

# 读取数据
df = pd.read_csv('train_data.csv')

label = df['Churn']
data = df.iloc[:, 1:-1]
X = data
# 数据预处理：将特征进行标准化
# scaler = StandardScaler()
# X = scaler.fit_transform(data)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.2, random_state=42)

# 创建并训练神经网络模型
model = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 输出结果
print("预测结果：", accuracy)
# 0.7

