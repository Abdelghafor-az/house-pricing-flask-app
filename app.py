from flask import Flask, request, render_template, url_for, jsonify

import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model pickle file
regmodel = pickle.load(open('reg_model.pkl', 'rb'))

# Load scaler pickle file
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # get data from postman request
    data = request.json['data']

    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    predicted_val = regmodel.predict(new_data) 

    return jsonify(predicted_val[0])

if __name__ == "__main__":
    app.run(debug=True)

    

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    
