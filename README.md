# HEALTH IA

HEALTH IA is an innovative application powered by artificial intelligence that aims to prevent cervical cancer in women by analyzing historical data of the mortality rate of this disease.

## How it works

Medical information is provided regarding: A- Hinselmann B -Citology C- Schiller Using advanced machine learning techniques, HEALT IA analyzes this data to generate a personalized risk profile for each user.

Additionally, the app offers personalized recommendations on screening, medical follow-up, and preventative measures that can help reduce the risk of developing cervical cancer. These recommendations are based on the latest medical guidelines and continuous analysis of real-time data to ensure the accuracy and effectiveness of the system.

## Files included

| File                                                                      | Details              |
| ------------------------------------------------------------------------- | -------------------- |
| [\__init_.py](app/__init__.py)                                            | is the main file and |
| Contains the endpoint with the file that represents the prediction model. |
| [random_forest_TF.joblib](/random_forest_TF.joblibra)                     |
| is the serialized file of the predictive model                            |

#### example:

```
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
```

## tools:

- Flask==3.0.3
- Flask-Cors==4.0.1
- numpy==1.26.0
- pandas==2.1.1
- pluggy==1.3.0
- scikit-learn==1.4.2

Athor:

- Daniel ruiz [Linkedin](https://www.linkedin.com/in/daniel-ruiz)
- Julian Andres Mendoza Castro [Linkedin](https://www.linkedin.com/in/julian-mendoza-castro/)
- Camilo Campaz [Linkedin](https://www.linkedin.com/in/camilo-campaz-jimenez/)
