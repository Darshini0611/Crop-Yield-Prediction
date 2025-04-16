import pickle
import numpy as np
from flask import Flask, render_template, request

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Define the threshold value
threshold = 77053.33

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = float(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # Prepare features for prediction
        features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]])
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features)[0]

        # Create a message based on the prediction
        if prediction > threshold:
            message = f"The predicted yield is {prediction:.2f}. This yield is good for cultivating {Item} in {Area}."
        else:
            message = f"The predicted yield is {prediction:.2f}. This yield may not be sufficient for optimal cultivation of {Item} in {Area}."

        return render_template('index.html', prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
