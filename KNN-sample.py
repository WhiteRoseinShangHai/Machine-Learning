#使用KNN算法分类数据
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier


#导入iris数据
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(iris_X, iris_Y, test_size=0.3)


#定义使用KNN算法
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
print(knn.score(X_test,Y_test)) #输出正确率
