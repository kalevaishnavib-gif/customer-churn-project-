# churn_model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load dataset
data = pd.read_csv('data/customer_churn.csv')

# 2. Preprocess data
# Convert 'Churn' to 0/1
data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})

# Encode categorical columns
le = LabelEncoder()
for col in ['gender', 'Contract', 'PaymentMethod']:
    data[col] = le.fit_transform(data[col])

# 3. Split data
X = data[['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']]
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save trained model
with open('model/churn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")