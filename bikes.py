
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


df = pd.read_csv('hour.csv')
y = df['cnt']

col_del = ['cnt', 'casual', 'registered', 'dteday', 'instant']
df.drop(col_del, axis=1, inplace=True)

transformers = [
    ['one_hot', OneHotEncoder(), ['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit']],
    ['scaler', RobustScaler(), ['temp', 'atemp', 'hum', 'windspeed']]
]
ct = ColumnTransformer(transformers, remainder='passthrough')
X = ct.fit_transform(df)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train,y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)

mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)

print(f'Mean_cnt {np.median(y)}')
print(f'Train: {mae_train}, Test: {mae_test}')