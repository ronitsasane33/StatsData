import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve, GridSearchCV, ShuffleSplit
import pickle

data = pd.read_csv('static/data files/BostonHousing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=0)


regressor = SGDRegressor()
regressor.fit(X_train, y_train)


pickle.dump(regressor, open('static/PKL files/housing/housingSGDRegressor.pkl','wb'))


