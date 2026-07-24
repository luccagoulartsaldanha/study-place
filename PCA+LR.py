import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib as plt
import pandas as pd


data = load_breast_cancer()
x = pd.DataFrame(data)
print(x)