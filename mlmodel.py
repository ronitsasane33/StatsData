import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("FuelConsumption.csv")


cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]


regressor = LinearRegression()

regressor.fit(x, y)

pickle.dump(regressor, open('model.pkl','wb'))


