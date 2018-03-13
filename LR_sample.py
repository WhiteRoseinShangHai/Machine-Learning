from sklearn.linear_model import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
X_data = loaded_data.data
Y_data = loaded_data.target

model = LinearRegression()
model.fit(X_data, Y_data)

X, Y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=10)
plt.scatter(X, Y)
plt.show()
