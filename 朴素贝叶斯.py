import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_data.csv')

x = data.iloc[:, 1:-1]
y = data['Churn']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.naive_bayes import GaussianNB

# 朴素贝叶斯
classifier = GaussianNB()

classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred_nb = classifier.predict(X_test)

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("Naive Bayes Accuracy:", accuracy_nb)

# acc:0.6560747663551402
