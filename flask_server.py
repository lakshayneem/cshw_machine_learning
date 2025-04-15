import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
import pickle
app = Flask(__name__)
CORS(app, origins=["https://cshw-frontend-fake-job-detection.onrender.com"])

print("✅ CORS configured for: https://cshw-frontend-fake-job-detection.onrender.com")

# Load model and preprocessors
model = tf.keras.models.load_model("fake_job_detection.h5")

tokenizer_description = joblib.load("tokenizer_description.pkl")
tokenizer_requirements = joblib.load("tokenizer_requirements.pkl")
tokenizer_company_profile = joblib.load("tokenizer_company_profile.pkl")
tokenizer_benefits = joblib.load("tokenizer_benefits.pkl")
one_hot_enc = joblib.load("one_hot_encoder.pkl")

# Ensure handle_unknown='ignore' is set in OneHotEncoder
one_hot_enc.handle_unknown = 'ignore'

# Expected feature size based on training
EXPECTED_FEATURES = 728

def preprocess_text(text, tokenizer):
    """Tokenize and pad text to max length 90."""
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=90, padding='post')


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive input data
        data = request.json
        print("\n🔹 Received Data:", data)  # Debugging Line

        # Process text fields
        company_profile = preprocess_text(data.get("company_profile", ""), tokenizer_company_profile)
        description = preprocess_text(data.get("description", ""), tokenizer_description)
        requirements = preprocess_text(data.get("requirements", ""), tokenizer_requirements)
        benefits = preprocess_text(data.get("benefits", ""), tokenizer_benefits)

        # Process categorical features
        categorical_values = pd.DataFrame(
            [[data.get("employment_type", "Other"),
              data.get("required_experience", "Not Applicable"),
              data.get("required_education", "Unspecified"),
              data.get("industry", "Other"),
              data.get("function", "Other")]],
            columns=["employment_type", "required_experience", "required_education", "industry", "function"]
        )

        # Encode categorical features
        categorical_encoded = one_hot_enc.transform(categorical_values)

        # Process boolean features
        boolean_features = np.array([[int(data.get("telecommuting", 0)), 
                                      int(data.get("has_company_logo", 1)), 
                                      int(data.get("has_questions", 0))]])

        # Concatenate all features
        X_input = np.hstack([company_profile, description, requirements, benefits, categorical_encoded, boolean_features])

        # Debugging info before prediction
        print("🔹 X_input shape:", X_input.shape)
        print("🔹 Any NaN in input:", np.isnan(X_input).any())
        print("🔹 Any Infinity in input:", np.isinf(X_input).any())
        print("🔹 Min:", np.min(X_input), "Max:", np.max(X_input))

        # Fix shape mismatch by padding if necessary
        current_features = X_input.shape[1]
        if current_features < EXPECTED_FEATURES:
            padding = np.zeros((1, EXPECTED_FEATURES - current_features))
            X_input = np.hstack([X_input, padding])

        # Ensure no NaN values
        X_input = np.nan_to_num(X_input)

        # Predict using model
        prediction = model.predict(X_input)
        print("🔹 Model Output:", prediction)  # Debugging

        confidence = float(prediction[0][0])

        # Generate response
        label = "Fake Job Posting" if confidence >= 0.5 else "Legitimate Job Posting"
        confidence_percent = confidence * 100 if confidence >= 0.5 else (1 - confidence) * 100

        print(f"✅ Prediction: {label}, Confidence: {confidence_percent:.2f}%")  # Debugging

        return jsonify({"prediction": label, "confidence": f"{confidence_percent:.2f}%"})

    except Exception as e:
        print("❌ Error:", str(e))  # Debugging
        return jsonify({"error": str(e)})



import re
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
# Load stopwords (excluding 'not')
stop_words = set(stopwords.words('english')) - {'not'}

# Load Tokenizer & Model
try:
    with open("tokenizer.pkl", "rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    model_sentiment = load_model("lstm_sentiment_model.h5")  # Load trained Bi-LSTM model
except FileNotFoundError:
    print("Error: Tokenizer or Model file not found!")
    tokenizer, model_sentiment = None, None

# Define max length (must match training)
MAX_LEN = 150  # Adjust based on training data

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()



def preprocessing_text(text):
    """Text cleaning: remove special chars, lemmatize, and handle negations."""
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    
    processed_words = []
    i = 0
    while i < len(text):
        if text[i] == "not" and i + 1 < len(text):
            processed_words.append(f"not_{text[i+1]}")
            i += 2
        elif text[i] not in stop_words:
            processed_words.append(lemmatizer.lemmatize(text[i]))
            i += 1
        else:
            i += 1

    return ' '.join(processed_words)

@app.route("/api/analyze", methods=["POST"])
def analyze_review():
    """API for Sentiment Analysis"""
    if not model_sentiment or not tokenizer:
        return jsonify({"error": "Model or Tokenizer not loaded"}), 500

    data = request.json
    review = data.get("review", "")

    if not review.strip():
        return jsonify({"error": "Review cannot be empty"}), 400

    # Preprocess text
    processed_review = preprocessing_text(review)

    if not processed_review:
        return jsonify({"error": "Processed review is empty"}), 400

    # Convert text to sequence & pad
    sequence = tokenizer.texts_to_sequences([processed_review])
    review_vector = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')

    # Predict sentiment
    prediction = model_sentiment.predict(review_vector)[0][0]  # Get probability

    response = {
        "positive": int(prediction >= 0.5),
        "score": round(float(prediction), 4)
    }

    return jsonify(response)



if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
