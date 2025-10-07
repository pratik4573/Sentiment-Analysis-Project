📝 Tweet Sentiment Analysis Project

Project Overview

This project performs Sentiment Analysis on Tweets using Natural Language Processing (NLP) and Machine Learning.

The project allows users to classify tweets or short texts into Positive, Neutral, or Negative sentiment. It also includes a Streamlit web app for interactive sentiment prediction.

11️⃣ Project Structure

Tweet-Sentiment-Analysis/
│
├── app.py                 # Streamlit web application

├── tweets_dataset.csv     # Sample dataset

├── sentiment_model.pkl    # Trained ML model

├── tfidf_vectorizer.pkl   # TF-IDF vectorizer

├── README.md              # Project documentation

└── requirements.txt       # Required Python packages

Key Steps:

1.Data Preparation: Created a balanced CSV dataset of 100 tweets with sentiments (Positive, Neutral, Negative).

2.Text Preprocessing: Lowercased text, removed URLs, mentions, hashtags, numbers, punctuation, and stopwords using NLTK.

3.Feature Extraction: Converted text into TF-IDF vectors.

4.Model Training: Used Logistic Regression with class_weight='balanced' to handle imbalanced classes. Split data into training/testing sets and evaluated accuracy.

5.Model Saving: Saved trained model and vectorizer with joblib.

6.Streamlit App:

    (a)Analyze single tweets or bulk CSV files.

    (b)Clean input, predict sentiment, and display results.

    (c)Option to download predictions as CSV.

7.Learnings:

    (a)NLP preprocessing, handling imbalanced datasets, TF-IDF features, ML model training, and web app deployment with Streamlit.
