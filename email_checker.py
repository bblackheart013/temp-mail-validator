import re
import joblib

# Load the model and vectorizer
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def is_valid_email_format(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def classify_domain(domain):
    domain_vectorized = vectorizer.transform([domain])
    prediction = model.predict(domain_vectorized)[0]
    probability = model.predict_proba(domain_vectorized)[0]
    return prediction, max(probability)

def check_email(email):
    if not is_valid_email_format(email):
        return False, "Invalid email format", 1.0

    domain = email.split('@')[-1]
    prediction, confidence = classify_domain(domain)
    
    if prediction == 0:
        return True, "Valid email domain", confidence
    else:
        return False, "Likely temporary or disposable email domain", confidence

if __name__ == "__main__":
    while True:
        email = input("Enter an email address (or 'quit' to exit): ")
        if email.lower() == 'quit':
            break
        
        is_valid, reason, confidence = check_email(email)
        if is_valid:
            print(f"This email appears to be valid. {reason} (Confidence: {confidence:.2f})")
        else:
            print(f"This email is likely invalid. {reason} (Confidence: {confidence:.2f})")

print("Thank you for using the email checker!")