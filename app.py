# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
# Load the regression model
classifier = pickle.load(open('rfr.pickle', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # '''
    # For rendering results on HTML GUI
    # '''
    age_of_car = int(request.form['age_of_car'])
    transmission = int(request.form['transmission'])
    mileage = int(request.form['mileage'])
    fuelType = int(request.form['fuelType'])
    tax = int(request.form['tax'])
    mpg = float(request.form['mpg'])
    engineSize = float(request.form['engineSize'])
    
    data = [[age_of_car,transmission,mileage,fuelType,tax,mpg,engineSize]]
    if (age_of_car == 0 and transmission == 0 and mileage == 0 and fuelType == 0 and tax ==0 and mpg == 0 and engineSize == 0):
        my_prediction = [0]
    else:
        my_prediction = classifier.predict(data)

    return render_template('output.html', prediction_price='${}'.format(round(my_prediction[0],2)))



if __name__ == "__main__":
    app.run(debug=True)
