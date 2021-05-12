import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
import pickle


data=pd.read_csv('static/data files/iris.csv')
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
data['species']= label_encoder.fit_transform(data['species']) 


X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

reg = SGDRegressor(x_train, y_train)


pickle.dump(reg, open('static/PKL files/iris/irisSGD.pkl','wb'))


