from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv('train_data.csv')

label = df['Churn']
data = df.iloc[:, 1:-1]
print(label.shape)
print(data.shape)
# 使用随机森林进行多分类
clf = RandomForestClassifier(random_state=42, criterion='entropy', max_depth=7, min_samples_leaf=5, min_samples_split=4, n_estimators=300)
# n_estimators： int 型，默认为 100。表示森林中树的数量，即有多少个基础决策树模型组成随机森林。该值越大，模型的拟合能力就越强，但是训练时间也会随之增加。
# criterion：字符串类型，可选参数，默认为 gini。表示决策树节点的分裂准则，可以是基尼系数或信息增益。一般来说，选择 Gini 指数计算速度较快，在实际生产环境中得到广泛应用。
# max_depth： int / None，表示决策树的最大深度。当参数为 None 时，每个节点会被扩展直到所有叶子都含有小于 min_samples_split 个样本，或者叶子节点仅由同一个类别组成。
# min_samples_split：int 、 float。表示一个内部节点需要多少个样本才能进一步细分，如果输入浮点数，则它代表以百分比形式的最小样本数。一般推荐使用默认值 2。
# min_samples_leaf：int 、 float。与min_samples_split 类似，它是指一个叶子节点所需要的最少样本数量或者比例，如果输入浮点数，则它代表以百分比形式的最小样本量。默认值为 1。
# max_features： int / float / {'auto', 'sqrt', 'log2'}，表示在每个节点处访问该特征的集合大小。以整数为参数时，意味着在每个节点处随机选择输入特征的子集并查找最佳划分，而 float 型参数则可以设置每次查找时包含特征的最大比例。指定“auto”或“sqrt”参数将使模型每次搜索的特征数等于总特征数的平方根，而指定 “log2” 参数将使模型每次搜索的特征数等于总特征数的对数。数据集中有大量特征时，应适当减小这个值以获得较好的分类效果。
# bootstrap：bool 型，默认 True。表示是否采用自助法进行有放回抽样（Bootstrapping samples）来训练决策树。该值为 True 时，表示用 Bootstrap 技术随机从原始样本中采样形成新的训练集。
# oob_score：bool 型，默认 False。表示是否使用袋外样本（Out-of-Bag samples）对模型进行评估。该参数设置为 True 可以提高模型的泛化能力。
# n_jobs：int 参数，表示在执行时要用到的核数量，如果设置为 -1，将使用所有的内核。设置这个参数可以减少训练时间。
# random_state：int 型或 None，默认 None。它是用于设置随机种子的超参数。当随机数生成器的初始状态相同时，它可以确保每次运行算法得到相同的结果。如果为 None，则每次生成随机数种子可能不同。


# params = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [4, 5, 6, 7],
#     'n_estimators': [100, 200, 300],
#     'min_samples_split': [4, 6, 8],
#     'min_samples_leaf': [3, 5, 7]
# }
# clf = GridSearchCV(clf, params, cv=3)
# clf.fit(data, label)
# print(clf.best_params_)


# 划分训练集和测试集

# {'criterion': 'entropy', 'max_depth': 7, 'min_samples_leaf': 5, 'min_samples_split': 4, 'n_estimators': 300}
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

clf.fit(data, label)

y_pred = clf.predict(X_test)
# 输出分类报告
print(classification_report(y_test, y_pred))

# 0.77