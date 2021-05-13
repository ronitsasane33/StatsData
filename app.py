import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#Iris
irisModel = pickle.load(open('static/PKL files/iris/irisModel.pkl', 'rb'))
irisDecisionTree = pickle.load(open('static/PKL files/iris/irisDecisionTree.pkl', 'rb'))
irisXGB = pickle.load(open('static/PKL files/iris/irisXGB.pkl', 'rb'))
# irisSGD = pickle.load(open('static/PKL files/iris/irisSGD.pkl', 'rb'))
irisGradientBoost = pickle.load(open('static/PKL files/iris/irisGradientBoost.pkl', 'rb'))

#emission
emissionModel = pickle.load(open('static/PKL files/emission/emissionModel.pkl', 'rb'))
emissionDecisionTree = pickle.load(open('static/PKL files/emission/emissionDecisionTree.pkl', 'rb'))
emissionXGB = pickle.load(open('static/PKL files/emission/emissionXGB.pkl', 'rb'))
emissionSGD = pickle.load(open('static/PKL files/emission/emissionSGD.pkl', 'rb'))
emissionGradientBoost = pickle.load(open('static/PKL files/emission/emissionGradientBoosting.pkl', 'rb'))

# Housing
housingLinearRegModel = pickle.load(open('static/PKL files/housing/housingModel.pkl', 'rb'))
housingDecisionTree = pickle.load(open('static/PKL files/housing/decisionTree.pkl', 'rb'))
housingXGB = pickle.load(open('static/PKL files/housing/housingXGB.pkl', 'rb')) 
housingSGD = pickle.load(open('static/PKL files/housing/housingSGDRegressor.pkl', 'rb')) 
housingGradientBoost= pickle.load(open('static/PKL files/housing/housingGradientBoost.pkl', 'rb')) 

# Page URLs
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/iris')
def iris():
    return render_template('iris.html')

@app.route('/o2emission')
def o2emission():
    return render_template('o2emission.html')

@app.route('/housing')
def housing():
    return render_template('housing.html')

@app.route('/coming_soon')
def coming_soon():
    return render_template('ComingSoon.html')

# URLS for processing Models

# Iris route
@app.route('/predictIris',methods=['POST'])
def predictIris():
    init_features = [x for x in request.form.values()]
    model_to_use = init_features[-1] 
    init_features = init_features[:4]
    init_features = [float(x) for x in init_features]
    final_features = [np.array(init_features)]

    if model_to_use == 'decision tree':
        prediction = irisDecisionTree.predict(final_features) 
    elif model_to_use == 'Gradient Boost':
        prediction = irisGradientBoost.predict(final_features) 
    elif model_to_use == "XGBoost":
        prediction = irisXGB.predict(final_features)
    else:
        prediction = irisModel.predict(final_features) 
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    return render_template('iris.html', prediction_text='Class of Flower: {}'.format(classes[round(abs(prediction[0]))]))


# CO2 Emission route
@app.route('/predictEmission',methods=['POST'])
def predictEmission():
    init_features = [x for x in request.form.values()]
    model_to_use = init_features[-1] 
    init_features = init_features[:3]
    init_features = [float(x) for x in init_features]
    final_features = [np.array(init_features)]

    if model_to_use == 'decision tree':
        prediction = emissionDecisionTree.predict(final_features) 
    elif model_to_use == 'Gradient Boost':
        prediction = emissionGradientBoost.predict(final_features) 
    elif model_to_use == "SGD":
        prediction = emissionSGD.predict(final_features)
    elif model_to_use == "XGBoost":
        prediction = emissionXGB.predict(final_features)
    else:
        prediction = emissionModel.predict(final_features) 

    return render_template('o2emission.html', prediction_text='Predicted CO2 Emission: {}'.format(round(abs(prediction[0]), 2))) 

# Housing route
@app.route('/predictHousing',methods=['POST'])
def predictHousing():
    init_features = [x for x in request.form.values()]
    model_to_use = init_features[-1] 
    init_features = init_features[:3]
    init_features = [float(x) for x in init_features]
    final_features = [np.array(init_features)]

    if model_to_use == 'decision tree':
        prediction = housingDecisionTree.predict(final_features) 
    elif model_to_use == 'Gradient Boost':
        prediction = housingGradientBoost.predict(final_features) 
    elif model_to_use == "SGD":
        prediction = housingSGD.predict(final_features)
    elif model_to_use == "XGBoost":
        prediction = housingXGB.predict(final_features)
    else:
        prediction = housingLinearRegModel.predict(final_features) 

    return render_template('housing.html', prediction_text='Predicted Price: ${}'.format(round(abs(prediction[0]), 2))) 
    

if __name__ == "__main__":
    app.run(debug=True)

    