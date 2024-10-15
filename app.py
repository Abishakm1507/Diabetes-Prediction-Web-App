import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # Load the scaler object

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    if request.method == 'POST':
        pregnancies = int(request.form['pregnancies'])
        glucose = int(request.form['Glucose'])
        blood_pressure = int(request.form['BloodPressure'])
        skin_thickness = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        # Create the data array with all features
        data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        
        # Scale the input data
        data_scaled = scaler.transform(data)
        
        # Predict the outcome
        my_prediction = model.predict(data_scaled)

        if my_prediction[0] == 0:
            output = "No Diabetes"
        else:
            output = "Diabetes"

    return render_template('index.html', prediction_text="Result: {}".format(output))

if __name__ == "__main__":
    flask_app.run(debug=True)
