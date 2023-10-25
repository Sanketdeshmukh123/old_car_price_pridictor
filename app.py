from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the regression model
classifier = pickle.load(open('rfr.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        age_of_car = int(request.form['age_of_car'])
        transmission = int(request.form['transmission'])
        mileage = int(request.form['mileage'])
        fuelType = int(request.form['fuelType'])
        tax = int(request.form['tax'])
        mpg = float(request.form['mpg'])
        engineSize = float(request.form['engineSize'])

        # Log the input data for debugging
        print("Input Data: age_of_car={}, transmission={}, mileage={}, fuelType={}, tax={}, mpg={}, engineSize={}".format(age_of_car, transmission, mileage, fuelType, tax, mpg, engineSize))

        data = [[age_of_car, transmission, mileage, fuelType, tax, mpg, engineSize]]
        my_prediction = classifier.predict(data)

        # Log the prediction for debugging
        print("Prediction: {}".format(my_prediction))

        return render_template('output.html', prediction_price='₹{}'.format(round(my_prediction[0], 2)))
    
    except Exception as e:
        error_message = "Error: " + str(e)
        return render_template('output.html', prediction_price=error_message)

if __name__ == "__main__":
    app.run(debug=True)
