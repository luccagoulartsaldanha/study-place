import pandas as pd
from matplotlib import pyplot as plt    
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv('500hits.csv')
data = data.drop(['PLAYER', 'CS'], axis=1)
x = data[['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB']]
y = data[['HOF']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain, ytrain.values.ravel())
model.predict(xtest)
print(confusion_matrix(ytest, model.predict(xtest)))