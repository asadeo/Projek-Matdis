from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

def train_models():
    data = pd.read_csv('student-mat.csv')

    # Tangani missing values untuk kolom numerik
    data.update(data.select_dtypes(include=[np.number]).median())
    data['age'].fillna(data['age'].median(), inplace=True)

    # Tangani missing values untuk kolom non-numerik
    for col in data.select_dtypes(exclude=[np.number]).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # Prepare features
    X = data[['studytime', 'absences', 'G1', 'G2', 'age', 'famsize', 'traveltime', 'failures', 'schoolsup', 'higher']]
    y_classification = (data['G3'] >= 13).astype(int)
    y_regression = data['G3']

    # Encode categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes

    # Split data
    X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_classification, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)

    # Train models
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train_class)

    regressor = RandomForestRegressor(random_state=42)
    regressor.fit(X_train, y_train_reg)

    return classifier, regressor

classifier, regressor = train_models()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    required_features = [
        'studytime', 'absences', 'G1', 'G2', 'age', 
        'famsize', 'traveltime', 'failures', 'schoolsup', 'higher'
    ]
    try:
        data = request.json

        # Periksa apakah ada missing values
        for feature in required_features:
            if feature not in data or data[feature] is None:
                return jsonify({"error": f"Missing feature: {feature}"})

        user_input_df = pd.DataFrame([data])
        user_input_df.fillna(user_input_df.median(), inplace=True)

        # Predictions
        prediction_class = classifier.predict(user_input_df)[0]
        prediction_reg = regressor.predict(user_input_df)[0]
        proba = classifier.predict_proba(user_input_df)[0][1]

        result = {
            "classification": "Pass" if prediction_class == 1 else "Fail",
            "regression": round(prediction_reg, 2),
            "probability": round(proba * 100, 2)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
