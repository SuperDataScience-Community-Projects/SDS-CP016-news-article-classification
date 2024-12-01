#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:22:45 2024

@author: rajatthakur
"""
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

about = """ 
Welcome to the Article Classification App, a powerful yet simple tool that classifies articles into predefined categories using machine learning.
**Key Features:**
* Article Classification: Paste any text, and the app predicts the category with confidence.
* Class Probabilities: View a breakdown of probabilities for all possible categories.
* Confidence Score: See how confident the model is about its prediction.

**Behind the Scenes:**
* The app uses a Naive Bayes classifier trained on a TF-IDF vectorized dataset.
* It leverages the NLTK library for text preprocessing, including removing stop words and tokenization.

**How to Use:**
* Enter your article or text in the input box.
* Click on Classify to see the predicted category and probabilities.
* Explore class probabilities visually in a bar chart.

**This app is ideal for quick text categorization tasks and can be used for news classification, content tagging, or educational purposes.**
"""
st.set_page_config(layout="wide", page_title="Article Classification App")
st.sidebar.title("About This App")
st.sidebar.write(about)



tfidf = joblib.load('tfidf_vectorizer.pkl')
nb_model = joblib.load('nb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


def predicted_label(article,model):
    processed_text = re.sub('[^a-zA-Z]', ' ', article).lower()
    stop_words = set(stopwords.words('english'))
    processed_text = ' '.join([word for word in processed_text.split() if word not in stop_words])
    text_tfidf = tfidf.transform([processed_text]).toarray()
    probabilities = model.predict_proba(text_tfidf)[0]
    predicted_class = model.predict(text_tfidf)[0]
    confidence = probabilities[predicted_class]
    return predicted_class, confidence, probabilities
    


def main():
    st.title("Article Classification App")
    #st.write("Welcome to article classification app. Enter an article below to classify it")
    
    input_text_placeholder = "Please enter some text to classify"
    article_input= st.text_area("Input article text", height=200,placeholder=(input_text_placeholder) ,label_visibility="visible")
 
    if st.button("Classify"):
            
     if article_input.strip():
            st.write("Processing your Article...") 
            numeric_label, confidence, probabilities = predicted_label(article_input, nb_model)
            readable_label = label_encoder.inverse_transform([numeric_label])[0]
            st.success(f"The predicted category by NB is: **{readable_label}** with **{confidence * 100:.2f}%** Confidence")
        

            st.write("Class Probabilities: ")
            class_labels = label_encoder.classes_
            st.bar_chart(pd.DataFrame(probabilities, index=class_labels, columns=["Probability"]))
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(confidence)
            
        
    


if __name__ == "__main__":
    main()
