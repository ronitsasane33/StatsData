import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
emissionModel = pickle.load(open('emissionModel.pkl', 'rb'))
irisModel = pickle.load(open('irisModel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/iris')
def iris():
    return render_template('iris.html')

@app.route('/o2emission')
def o2emission():
    return render_template('o2emission.html')

# CO2 Emission route
@app.route('/predictEmission',methods=['POST'])
def predictEmission():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = emissionModel.predict(final_features) 
    return render_template('o2emission.html', prediction_text='Predicted Class: {}'.format(prediction)) 

# Iris route
@app.route('/predictIris',methods=['POST'])
def predictIris():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = irisModel.predict(final_features) 
    return render_template('iris.html', prediction_text='Predicted Class: {}'.format(prediction)) 

if __name__ == "__main__":
    app.run(debug=True)

    