import os
import re
import joblib
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from middleware import CORSMiddleware

app.wsgi_app = CORSMiddleware(app.wsgi_app)

# Flask app initialization
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# === Load Models and Tokenizers ===
# Fake Job Detection
model_job = tf.keras.models.load_model("fake_job_detection.h5")
tokenizer_description = joblib.load("tokenizer_description.pkl")
tokenizer_requirements = joblib.load("tokenizer_requirements.pkl")
tokenizer_company_profile = joblib.load("tokenizer_company_profile.pkl")
tokenizer_benefits = joblib.load("tokenizer_benefits.pkl")
one_hot_enc = joblib.load("one_hot_encoder.pkl")
one_hot_enc.handle_unknown = 'ignore'
EXPECTED_FEATURES = 728

# Sentiment Analysis
try:
    with open("tokenizer.pkl", "rb") as tokenizer_file:
        tokenizer_sentiment = pickle.load(tokenizer_file)
    model_sentiment = tf.keras.models.load_model("lstm_sentiment_model.h5")
except FileNotFoundError:
    print("Error: Tokenizer or Model file not found!")
    tokenizer_sentiment, model_sentiment = None, None

# === Helpers ===

def preprocess_text(text, tokenizer):
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=90, padding='post')

def preprocessing_review(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    stop_words = set(stopwords.words('english')) - {'not'}
    lemmatizer = WordNetLemmatizer()

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

# === Routes ===

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        company_profile = preprocess_text(data.get("company_profile", ""), tokenizer_company_profile)
        description = preprocess_text(data.get("description", ""), tokenizer_description)
        requirements = preprocess_text(data.get("requirements", ""), tokenizer_requirements)
        benefits = preprocess_text(data.get("benefits", ""), tokenizer_benefits)

        categorical_values = pd.DataFrame(
            [[data.get("employment_type", "Other"),
              data.get("required_experience", "Not Applicable"),
              data.get("required_education", "Unspecified"),
              data.get("industry", "Other"),
              data.get("function", "Other")]],
            columns=["employment_type", "required_experience", "required_education", "industry", "function"]
        )
        categorical_encoded = one_hot_enc.transform(categorical_values)

        boolean_features = np.array([[int(data.get("telecommuting", 0)),
                                      int(data.get("has_company_logo", 1)),
                                      int(data.get("has_questions", 0))]])

        X_input = np.hstack([company_profile, description, requirements, benefits, categorical_encoded, boolean_features])

        current_features = X_input.shape[1]
        if current_features < EXPECTED_FEATURES:
            padding = np.zeros((1, EXPECTED_FEATURES - current_features))
            X_input = np.hstack([X_input, padding])
        X_input = np.nan_to_num(X_input)

        prediction = model_job.predict(X_input)
        confidence = float(prediction[0][0])

        label = "Fake Job Posting" if confidence >= 0.5 else "Legitimate Job Posting"
        confidence_percent = confidence * 100 if confidence >= 0.5 else (1 - confidence) * 100

        return jsonify({"prediction": label, "confidence": f"{confidence_percent:.2f}%"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_review():
    if not model_sentiment or not tokenizer_sentiment:
        return jsonify({"error": "Model or Tokenizer not loaded"}), 500

    data = request.json
    review = data.get("review", "")

    if not review.strip():
        return jsonify({"error": "Review cannot be empty"}), 400

    processed_review = preprocessing_review(review)
    if not processed_review:
        return jsonify({"error": "Processed review is empty"}), 400

    sequence = tokenizer_sentiment.texts_to_sequences([processed_review])
    review_vector = pad_sequences(sequence, maxlen=150, padding='post')
    prediction = model_sentiment.predict(review_vector)[0][0]

    response = {
        "positive": int(prediction >= 0.5),
        "score": round(float(prediction), 4)
    }

    return jsonify(response)

# === Run the app ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
