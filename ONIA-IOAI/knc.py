from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("football.csv")
x = data["away_score"]
y = data["attendance"]
plt.plot(x, y)
plt.show()




