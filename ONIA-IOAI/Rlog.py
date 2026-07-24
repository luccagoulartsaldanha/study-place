from pydataset import data
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
titanic = data('titanic')
titanic = pd.get_dummies(titanic, drop_first=True)
xtrain, xtest, ytrain, ytest = train_test_split(titanic.drop('survived_yes', axis=1), titanic['survived_yes'], test_size=0.2, random_state=42)
log = LogisticRegression()
log.fit(xtrain, ytrain)
print(log.score(xtest, ytest))