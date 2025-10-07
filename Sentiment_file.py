import streamlit as st
import pandas as pd
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# --- Download stopwords ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# --- Load Model and Vectorizer ---
try:
    model = joblib.load("C:\\Users\\15jan\\OneDrive\\build\\Desktop\\DA Project\\dataset\\sentiment_model.pkl")
    vectorizer = joblib.load("C:\\Users\\15jan\\OneDrive\\build\\Desktop\\DA Project\\dataset\\tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.warning("⚠️ Model or vectorizer not found. Train your model first.")
    st.stop()

# --- Text Cleaning Function ---
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)         # remove mentions
    text = re.sub(r"#\w+", "", text)         # remove hashtags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)          # remove numbers
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# --- Streamlit App Layout ---
st.set_page_config(page_title="Tweet Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Tweet Sentiment Analysis App")
st.write("Type a tweet or upload a CSV file to analyze sentiment (Positive, Neutral, Negative).")

# --- Option 1: Single Tweet Input ---
st.subheader("Analyze Single Tweet")
user_input = st.text_area("Enter a tweet here:", height=150, placeholder="Type something like 'I love this app!'...")

if st.button("🔍 Analyze Single Tweet"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        cleaned = clean_text(user_input)
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        st.success(f"Predicted Sentiment: {pred.upper()}")

# --- Option 2: CSV Upload ---
st.subheader("Analyze Multiple Tweets from CSV")
uploaded_file = st.file_uploader("Upload CSV file with a column 'text'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("❌ CSV must contain a column named 'text'.")
    else:
        # Clean and predict
        df['clean_text'] = df['text'].apply(clean_text)
        vect = vectorizer.transform(df['clean_text'])
        df['sentiment'] = model.predict(vect)
        st.write("✅ Predictions:")
        st.dataframe(df[['text', 'sentiment']])
        
        # Optionally save predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv,
            file_name="predicted_tweets.csv",
            mime="text/csv"
        )


