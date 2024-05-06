from flask import Flask, redirect, url_for, request
import os
from joblib import load
import sklearn as sk
from sklearn import tree , metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

def create_app():
    app = Flask(__name__)

    # Initialize Flask extensions here

    # Load the model
    
    # Register blueprints here

    @app.route('/test')
    def test_page():
        return 'Test page'

    def transformData(data):
      modified_data = {}
      for key, value in data.items():
        modified_data[key + '_0'] = 1 - int(value)  # 0 si True, 1 si False
        modified_data[key + '_1'] = int(value)  # 1 si True, 0 si False
      return modified_data
      
    @app.route('/predict', methods=['POST'])
    def predict():
      if request.method == 'POST':
        data = request.get_json()
        transformed_data = transformData(data)
        
        df = pd.DataFrame(transformed_data, index=[0])
        model = load('random_forest_TF.joblib')
       
        # Make a prediction
        processed_data = minmax_scale.fit_transform(df)
        prediction = model.predict(processed_data)
        # Return the prediction
      return str(prediction)
      
    def getAccuracy():
      model = load('random_forest_TF.joblib')
      X_test = pd.read_csv('X_test.csv')
      y_test = pd.read_csv('y_test.csv')
      y_pred = model.predict(X_test)
      return metrics.accuracy_score(y_test, y_pred)
    return app