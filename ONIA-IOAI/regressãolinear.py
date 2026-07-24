import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('testeRL.csv')
x = data[["SqFt"]]
y = data[["Price"]]

plt.scatter(x, y)
plt.show()
model = LinearRegression()
model.fit(x, y)
plt.scatter(x, y)
plt.plot(x, model.predict(x), color='red')
plt.show()


