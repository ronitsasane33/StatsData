import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

data=pd.read_csv('static/data files/iris.csv')
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
data['species']= label_encoder.fit_transform(data['species']) 

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
regressor = LogisticRegression()
regressor.fit(x_train, y_train) #Training the model

pickle.dump(regressor, open('static/PKL files/iris/irisModel.pkl','wb'))

