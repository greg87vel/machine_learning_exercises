from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

n = 100
X = np.random.random(size=(n, 5))
y = np.random.choice(['si', 'no'], size=n)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

model = MLPClassifier(hidden_layer_sizes=[500, 500], verbose=2)
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print(f'Train_acc: {acc_train} \n Test_acc: {acc_test}')