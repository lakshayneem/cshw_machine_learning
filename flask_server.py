from flask import Flask, request, jsonify
from flask_cors import CORS
import os, re, joblib, pickle, numpy as np, pandas as pd, nltk, tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize app
app = Flask(__name__)
CORS(app)

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load models/tokenizers
model_job = tf.keras.models.load_model("fake_job_detection.h5")
tokenizer_description = joblib.load("tokenizer_description.pkl")
tokenizer_requirements = joblib.load("tokenizer_requirements.pkl")
tokenizer_company_profile = joblib.load("tokenizer_company_profile.pkl")
tokenizer_benefits = joblib.load("tokenizer_benefits.pkl")
one_hot_enc = joblib.load("one_hot_encoder.pkl")
one_hot_enc.handle_unknown = 'ignore'
EXPECTED_FEATURES = 728

try:
    with open("tokenizer.pkl", "rb") as f:
        tokenizer_sentiment = pickle.load(f)
    model_sentiment = tf.keras.models.load_model("lstm_sentiment_model.h5")
except:
    tokenizer_sentiment, model_sentiment = None, None

# Helper functions
def preprocess_text(text, tokenizer):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=90, padding='post')

def preprocessing_review(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower().split()
    stop_words = set(stopwords.words('english')) - {'not'}
    lemmatizer = WordNetLemmatizer()
    words, i = [], 0
    while i < len(text):
        if text[i] == "not" and i + 1 < len(text):
            words.append(f"not_{text[i+1]}")
            i += 2
        elif text[i] not in stop_words:
            words.append(lemmatizer.lemmatize(text[i]))
            i += 1
        else: i += 1
    return ' '.join(words)

# Routes
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        company_profile = preprocess_text(data.get("company_profile", ""), tokenizer_company_profile)
        description = preprocess_text(data.get("description", ""), tokenizer_description)
        requirements = preprocess_text(data.get("requirements", ""), tokenizer_requirements)
        benefits = preprocess_text(data.get("benefits", ""), tokenizer_benefits)

        cat_vals = pd.DataFrame([[data.get("employment_type", "Other"),
                                  data.get("required_experience", "Not Applicable"),
                                  data.get("required_education", "Unspecified"),
                                  data.get("industry", "Other"),
                                  data.get("function", "Other")]],
                                columns=["employment_type", "required_experience", "required_education", "industry", "function"])
        cat_encoded = one_hot_enc.transform(cat_vals)

        bools = np.array([[int(data.get("telecommuting", 0)),
                           int(data.get("has_company_logo", 1)),
                           int(data.get("has_questions", 0))]])

        X = np.hstack([company_profile, description, requirements, benefits, cat_encoded, bools])
        if X.shape[1] < EXPECTED_FEATURES:
            X = np.hstack([X, np.zeros((1, EXPECTED_FEATURES - X.shape[1]))])
        X = np.nan_to_num(X)

        prediction = model_job.predict(X)
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
    processed = preprocessing_review(review)
    if not processed:
        return jsonify({"error": "Processed review is empty"}), 400
    seq = tokenizer_sentiment.texts_to_sequences([processed])
    vector = pad_sequences(seq, maxlen=150, padding='post')
    score = model_sentiment.predict(vector)[0][0]
    return jsonify({"positive": int(score >= 0.5), "score": round(float(score), 4)})

# Run
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
