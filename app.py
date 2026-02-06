# app.py
from flask import Flask, render_template, request
import pickle
import pandas as pd

# 1. Initialize Flask app
app = Flask(__name__)

# 2. Load trained model
with open('model/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 3. Home route – display form
@app.route('/')
def home():
    return render_template('index.html')

# 4. Predict route – handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gender = request.form['gender']
    senior = int(request.form['senior'])
    tenure = int(request.form['tenure'])
    monthly = float(request.form['monthly_charges'])
    total = float(request.form['total_charges'])
    contract = request.form['contract']
    payment = request.form['payment_method']

    # Convert categorical values same as in training
    gender = 1 if gender == 'Male' else 0
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_map = {'Electronic check':0, 'Mailed check':1, 'Bank transfer (automatic)':2, 'Credit card (automatic)':3}
    
    contract = contract_map[contract]
    payment = payment_map[payment]

    # Create DataFrame for prediction
    input_data = pd.DataFrame([[gender, senior, tenure, monthly, total, contract, payment]],
                              columns=['gender','SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Contract','PaymentMethod'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = 'Yes' if prediction == 1 else 'No'

    return render_template('result.html', prediction=result)

# 5. Run app
if __name__ == '__main__':
    app.run(debug=True)