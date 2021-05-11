import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("iris.csv")


cdf = df[['sepal_length','sepal_width','petal_length','petal_width']]

x = cdf.iloc[:, :4]
y = cdf.iloc[:, -1]


regressor = LinearRegression()

regressor.fit(x, y)

pickle.dump(regressor, open('irisModel.pkl','wb'))

