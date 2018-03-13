from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=4)
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    loss = -cross_val_score(knn, X, Y, cv=10, scoring='mean_squared_error') # for regression
    scores = cross_val_score(knn, X, Y, cv=10, scoring='accuracy') #for classification
    k_scores.append(scores.mean()) #loss.mean()
plt.plot(k_range, k_scores)
plt.show()
