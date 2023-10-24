from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


ds = load_wine()

X = ds['data'][:, [4, 7]]

scaler = QuantileTransformer()
X = scaler.fit_transform(X)

#df = pd.DataFrame(X, columns=['magnesium', 'phenols'])
#g = sns.scatterplot(data=df, x='magnesium', y='phenols')
#plt.show()

X = ds['data']
y = ds['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(p_train, y_train)
acc_test = accuracy_score(p_test, y_test)

print(f'Train: {acc_train}, Test: {acc_test}')


X = ds['data']
y = ds['target']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

model2 = KNeighborsClassifier()
model2.fit(X_train, y_train)

p_train = model2.predict(X_train)
p_test = model2.predict(X_test)

acc_train = accuracy_score(p_train, y_train)
acc_test = accuracy_score(p_test, y_test)

print(f'Train: {acc_train}, Test: {acc_test}')

