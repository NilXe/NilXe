from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

# 加载数据集
data = pd.read_csv('train_data.csv')

x = data.iloc[:, 1:-1]
y = data['Churn']


# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# # 定义超参数候选值
# param_grid = {'kernel': ['linear', 'f', 'poly'],
#               'C': [0.1, 1, 10],
#               'degree': [2, 3, 4]}
#
# # 创建 SVM 分类器和 GridSearchCV 对象
clf = svm.SVC(kernel='poly', C=1, degree=2)
# clf = GridSearchCV(svc, param_grid)

# 运行网格搜索算法
clf.fit(X_train, y)

p = clf.predict()
# 打印最佳超参数组合和得分
# print("Best parameters:", clf.best_params_)
# print("Best score:", clf.best_score_)
# 0.7475687820742593
