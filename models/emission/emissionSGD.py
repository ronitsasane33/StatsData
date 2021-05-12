import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv('static/data files/FuelConsumption.csv')
cdf = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
features = cdf.iloc[:, :3]
emissions = cdf.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(features, emissions, test_size=0.2, random_state=0)



from sklearn.metrics import make_scorer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import learning_curve, GridSearchCV, ShuffleSplit

regressor = SGDRegressor()
regressor.fit(X_train, y_train)


pickle.dump(regressor, open('static/PKL files/emission/emissionSGD.pkl','wb'))


