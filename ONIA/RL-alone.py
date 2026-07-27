from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

data = pd.read_csv("pokemon.csv")
data = data.query("Type1.isin(('Grass', 'Electric'))")
X = data[["Attack", "Speed", "Sp. Atk", "Sp. Def", "Total"]]
y = (data["Type1"] == "Grass")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=2)

model.fit(X_train, y_train)
plot_tree(model, feature_names=["Attack", "Speed", "Sp. Atk", "Sp. Def", "Total"], class_names=["Electric", "Grass"])
plt.show()
print(model.score(X_train, y_train))
model.predict(X_test)
print(model.score(X_test, y_test))