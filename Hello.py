import numpy
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

np.random.seed(2)

dataset = load_boston()

X = dataset['data']
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

mae_train= mean_absolute_error(y_train, p_train)
mae_test= mean_absolute_error(y_test, p_test)
print('MAE train', mae_train)
print('MAE test', mae_test)

