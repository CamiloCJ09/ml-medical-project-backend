from flask import Flask, redirect, url_for, request
import os
from joblib import load
import sklearn as sk
from sklearn import tree , metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
from flask_cors import CORS



def create_app():
    app = Flask(__name__)
    CORS(app, origins=["http://3.94.98.244", "*"])
    # Initialize Flask extensions here

    # Load the model
    
    # Register blueprints here

    @app.route('/test')
    def test_page():
        return 'Test page'

    def transformData(data):
      modified_data = {}
      for key, value in data.items():
        modified_data[key + '_0'] = not value  # 0 si True, 1 si False
        modified_data[key + '_1'] = value  # 1 si True, 0 si False
      return modified_data
      
    @app.route('/predict', methods=['POST'])
    def predict():
      if request.method == 'POST':
        data = request.get_json()
        
        transformed_data = transformData(data)
        
        
        df = pd.DataFrame(transformed_data, index=[1])
        model = load('random_forest_TF.joblib')
        
        prediction = model.predict(df)
        # Return the prediction
      return str(prediction[0])
    
    return app