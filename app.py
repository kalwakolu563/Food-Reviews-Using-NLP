 # Save this code in a file called app.py
import streamlit as st
from reviews import  clean_text , sia, vader_sentiment

st.title("Product Review Sentiment Analysis")
st.write("Enter a product review below:")

# Input box for user
review = st.text_area("Product Review:")

if st.button("Analyze Sentiment"):
    if review:
        cleaned_review = clean_text(review)
        score = sia.polarity_scores(cleaned_review)['compound']
        sentiment = vader_sentiment(score)
        st.write(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.write("Please enter a review.")

# to run this file : streamlit run app.py

