import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
# Setup
data = pd.read_csv("pokemon.csv")
data = data.query("Type1.isin(('Grass', 'Electric'))")
X = data[["Attack", "Speed", "Sp. Atk", "Sp. Def", "Total"]]
Y = (data["Type1"] == "Grass")

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=42)


# Decision Tree
max_depth = 2
tree = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, Y_train)

# Run
#plot_tree(tree, feature_names=["Attack", "Speed", "Sp. Atk", "Sp. Def", "Total"], class_names=["Electric", "Grass"])


# Predict
prediction = tree.predict(X_train)
answer = tree.predict(X_test)
#print(f"Max Depth: {max_depth}")
#print(f"Accuracy Train Score: {accuracy_score(Y_train, prediction)}")
#print(f"Precision Train Score: {precision_score(Y_train, prediction)}")
#print(f"Recall Train Score: {recall_score(Y_train, prediction)}")
#print(f"F1 Train Score: {f1_score(Y_train, prediction)}")
#print(f"Accuracy Test Score: {accuracy_score(Y_test, answer)}")
#print(f"Recall Test Score: {recall_score(Y_test, answer)}")
#print(f"F1 Test Score: {f1_score(Y_test, answer)}")
listaf1 = [f1_score(Y_test, answer)] 
listamaxdx = [max_depth]
while max_depth <= 10:
    tree = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, Y_train)
    answer = tree.predict(X_test)
    print(f"Max Depth: {max_depth}")
    print(f"F1 Test Score: {f1_score(Y_test, answer)}")
    listaf1 = listaf1 + [f1_score(Y_test, answer)]
    listamaxdx = listamaxdx + [max_depth]
    max_depth += 1


print(listaf1)
print(listamaxdx)


f1yt = listaf1
maxdx = listamaxdx

plt.plot(maxdx, f1yt, marker='o', linestyle='-')


plt.show()