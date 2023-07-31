# main.py
import pickle
from flask import Flask, request, render_template
import pandas as pd
from ML_Model import predict_mpg, preprocess_origin_cols, pipeline_transformer, CustomAttrAdder, num_pipeline_transformer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


app = Flask(__name__)

# Load the pre-trained machine learning model from model.bin using pickle
with open('model.bin', 'rb') as f_in:
    final_model = pickle.load(f_in)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', value="")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # Get the form data from the request
        cylinders = float(request.form['Cylinders'])
        displacement = float(request.form['Displacement'])
        horsepower = float(request.form['Horsepower'])
        weight = float(request.form['Weight'])
        acceleration = float(request.form['Acceleration'])
        model_year = float(request.form['Model_Year'])
        origin = float(request.form['Origin'])

        # Create a dictionary with the form data
        vehicle_config = pd.DataFrame({
            'Cylinders': [cylinders],
            'Displacement': [displacement],
            'Horsepower': [horsepower],
            'Weight': [weight],
            'Acceleration': [acceleration],
            'Model Year': [model_year],
            'Origin': [origin]
        })

        # Apply data preprocessing using pipeline_transformer from ML_Model.py
       # pipeline = pipeline_transformer(data)
       # prepared_df = pipeline.transform(data)

        # Call the predict_mpg function to get the predicted MPG
        prediction = predict_mpg(vehicle_config, final_model)
        # Retrieve the predicted value from the prediction list
        predicted_mpg = prediction[0]

        return render_template('index.html', value=predicted_mpg)

if __name__ == '__main__':
    app.run(debug=True)
