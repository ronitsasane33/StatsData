import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#Iris
irisModel = pickle.load(open('static/PKL files/iris/irisModel.pkl', 'rb'))

#emission
emissionModel = pickle.load(open('static/PKL files/emission/emissionModel.pkl', 'rb'))

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

# URLS for processing Models

# Iris route
@app.route('/predictIris',methods=['POST'])
def predictIris():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = irisModel.predict(final_features) 
    return render_template('iris.html', prediction_text='Predicted Class: {}'.format(abs(prediction[0])))

# CO2 Emission route
@app.route('/predictEmission',methods=['POST'])
def predictEmission():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = emissionModel.predict(final_features) 
    return render_template('o2emission.html', prediction_text='Predicted Class: {}'.format(abs(prediction[0])))

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

    return render_template('housing.html', prediction_text='Predicted Class: {}'.format(abs(prediction[0]))) 
    

if __name__ == "__main__":
    app.run(debug=True)

    