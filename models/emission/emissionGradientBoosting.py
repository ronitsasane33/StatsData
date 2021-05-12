import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


data = pd.read_csv('static/data files/FuelConsumption.csv')
cdf = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
features = cdf.iloc[:, :3]
emissions = cdf.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(features, emissions, test_size=0.2, random_state=0)


from sklearn.metrics import r2_score
def performance_metric(y_true, y_predict):    
    score = r2_score(y_true,y_predict)
    return score

from sklearn.metrics import make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import learning_curve, GridSearchCV, ShuffleSplit
def fit_model(X, y):
    
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.1, train_size=None, random_state=None)

    regressor = GradientBoostingRegressor(random_state=0)

    params = {'max_depth':list(range(1,11))}

    scoring_fnc = make_scorer(performance_metric)

    grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)

    grid = grid.fit(X, y)

    return grid.best_estimator_

reg = fit_model(X_train, y_train)


pickle.dump(reg, open('static/PKL files/emission/emissionGradientBoosting.pkl','wb'))


