import pandas as pd
import numpy as np
from scipy import sparse
import re
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
data = pd.read_csv('email_domains.csv')

# Function to preprocess domain
def preprocess_domain(domain):
    return re.sub(r'[^a-zA-Z0-9.-]', '', domain.lower())

# Preprocess domains
data['processed_domain'] = data['domain'].apply(preprocess_domain)

# Split the data into features and labels
X = data['processed_domain']
y = data['label']

# Use TfidfVectorizer for feature extraction
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 5))
X_vectorized = vectorizer.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize and train the RandomForestClassifier with GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
model = grid_search.best_estimator_

# Evaluate the model
y_pred = model.predict(X_test)

# Evaluate Random Forest
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Best parameters:", grid_search.best_params_)
print("Model Accuracy:", accuracy * 100)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save the model and vectorizer for later use
joblib.dump(model, 'rf_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Test with example domain inputs
while True:
    domain_input = input("Enter an email domain (or 'quit' to exit): ")
    if domain_input.lower() == 'quit':
        break
    processed_input = preprocess_domain(domain_input)
    input_vectorized = vectorizer.transform([processed_input])
    prediction = model.predict(input_vectorized)
    probability = model.predict_proba(input_vectorized)[0]
    
    if prediction[0] == 0:
        print(f"The domain '{domain_input}' is likely a valid email domain (Confidence: {probability[0]:.2f}).")
    else:
        print(f"The domain '{domain_input}' is likely a disposable or invalid email domain (Confidence: {probability[1]:.2f}).")

print("Thank you for using the email domain checker!")