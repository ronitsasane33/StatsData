import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/iris')
def iris():
    return render_template('iris.html')

@app.route('/o2emission')
def o2emission():
    return render_template('o2emission.html')

@app.route('/predict',methods=['POST'])
def predict():
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]
    prediction = model.predict(final_features) 
    return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction)) 

if __name__ == "__main__":
    app.run(debug=True)

    