# Import necessary libraries
import pandas as pd
import streamlit as st  # Modified
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vectorizer_file:
    model = pickle.load(model_file)
    tfidf = pickle.load(vectorizer_file)

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Text area for user input
input_sms = st.text_area('Enter message')

# Button to trigger prediction
if st.button('Predict'):
    # Preprocess the user input
    ps = PorterStemmer()  # Modified: Moved here
    text = input_sms.lower()  # Modified: Lowercasing
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english') and word not in set(string.punctuation)]
    tokens = [ps.stem(word) for word in tokens]
    transformed_sms = " ".join(tokens)
    # Vectorize the processed text
    vector_input = tfidf.transform([transformed_sms])
    # Predict using the loaded model
    result = model.predict(vector_input)[0]
    # Display prediction result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
