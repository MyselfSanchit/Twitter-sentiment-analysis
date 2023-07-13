import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the saved model
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Preprocessing steps
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Remove stopwords and lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    
    # Join the words back into a string
    processed_text = ' '.join(words)
    
    return processed_text

def predict_sentiment(tweet):
    # Preprocess the tweet
    processed_tweet = preprocess_text(tweet)
    
    # Transform the preprocessed tweet using the TF-IDF vectorizer
    transformed_tweet = vectorizer.transform([processed_tweet])
    
    # Make the sentiment prediction
    prediction = model.predict(transformed_tweet)[0]
    
    # Map the prediction label to sentiment category
    sentiment = {-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'}[prediction]
    
    # Get the probabilities for each class
    probabilities = model.predict_proba(transformed_tweet)
    
    # Get the index of the predicted sentiment category
    sentiment_index = model.classes_.tolist().index(prediction)
    
    # Get the probability for the predicted sentiment
    sentiment_probability = probabilities[0][sentiment_index]
    
    return sentiment, sentiment_probability

# Set page title
st.title("Twitter Sentiment Analysis")

# Create input text box for the user to enter a tweet
tweet_input = st.text_input("Enter a tweet:")

# Create a button to trigger sentiment prediction
if st.button("Predict Sentiment"):
    # Perform sentiment prediction
    sentiment, probability = predict_sentiment(tweet_input)
    
    # Display the sentiment and its probability
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Probability: {probability:.4f}")
