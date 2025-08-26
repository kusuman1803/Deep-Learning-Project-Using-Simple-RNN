#Step 1: Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

##Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

##Load the pretrained model with ReLU activation function
model = load_model("simplernn_imdb_model.keras")
model.summary()

##Step 2: Helper function to decode the review back to words
def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

##Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    # Convert tokens to their corresponding integer indices
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # 2 is for unknown words
    # Pad the sequence to ensure it has the same length as the training data
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

##tep 3 : Prediction function
def predict_sentiment(review):
    processed_review = preprocess_text(review)
    prediction = model.predict(processed_review)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

##Streamlit app to get user input and display prediction
import streamlit as st
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (Positive/Negative):")
user_input = st.text_area("Movie Review")

if st.button("Classify"):
    preprocess_input = preprocess_text(user_input)

    prediction = model.predict(preprocess_input)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    
    #Display the result
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Prediction Score: {prediction[0][0]}")
else:

    st.write("Please enter a movie review.")    
