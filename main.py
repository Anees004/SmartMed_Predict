import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify
import joblib

# Load and preprocess dataset
def load_dataset():
    df = pd.read_csv('dataset.csv')  # Replace with actual path to dataset
    df = df[['Disease', 'Symptom_1', 'Symptom_2', 'Symptom_3']]  # Using 3 symptoms for simplicity
    df.dropna(inplace=True)
    
    # Combine symptoms into one feature
    df['Symptoms'] = df['Symptom_1'].astype(str) + ' ' + df['Symptom_2'].astype(str) + ' ' + df['Symptom_3'].astype(str)
    
    return df

# Preprocessing and model training
def train_model():
    df = load_dataset()
    
    X = df[['Symptoms']]
    y = df['Disease']
    
    # Convert symptoms to numerical form
    vectorizer = CountVectorizer()
    X_vectorized = vectorizer.fit_transform(X['Symptoms'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
    
    # Train a decision tree model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    # Save the model and vectorizer
    joblib.dump(model, 'pathology_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

# Flask API
app = Flask(__name__)
model = joblib.load('pathology_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = ' '.join(data['symptoms'])  # Combine symptoms into one string
    
    # Transform symptoms to the vectorized form
    symptoms_vectorized = vectorizer.transform([symptoms]).toarray()
    
    prediction = model.predict(symptoms_vectorized)
    
    return jsonify({'predicted_pathology': prediction[0]})

if __name__ == '__main__':
    # Uncomment the next line if the model hasn't been trained yet
    train_model()
    
    app.run(debug=True)
