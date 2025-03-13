# Food-Reviews-Using-NLP
Amazon food reviews to determine the sentiment analysis of the reviews using NLP

Overview
This project focuses on sentiment analysis of Amazon food reviews using Natural Language Processing (NLP) techniques. The goal is to classify reviews as positive, negative, or neutral, helping users understand customer sentiment towards various food products.

We utilize machine learning models and NLP techniques to preprocess text, extract features, and predict sentiment. The project includes exploratory data analysis (EDA) to visualize review distributions and sentiment trends.

Additionally, we have integrated Streamlit, a Python-based web framework, to create an interactive web application where users can input text and receive real-time sentiment analysis.

This project is useful for businesses, consumers, and researchers who want to gain insights into customer opinions and enhance product feedback analysis.

Files and Structure
app.py: Uses Streamlit for sentiment analysis of reviews.


import streamlit as st
from reviews import clean_text, sia, vader_sentiment
eda.py: Performs exploratory data analysis using matplotlib and seaborn.


import matplotlib.pyplot as plt
import seaborn as sns
reviews.py: Implements sentiment analysis using a logistic regression model.


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
test.py: Testing file to verify updates and changes.

requirements.txt: Contains necessary libraries for environment setup.


Create a virtual environment (optional but recommended):
python -m venv env

Activate the virtual environment:
On Windows:
.\env\Scripts\activate

On macOS and Linux:
source env/bin/activate

Install dependencies:
pip install -r requirements.txt

Running the Project
To run the project, follow these steps:

EDA (Exploratory Data Analysis):
python eda.py

Streamlit App:
streamlit run app.py

Sentiment Analysis:
python reviews.py


Usage
1️⃣ Running the Sentiment Analysis

Using Streamlit Web App
Run the following command to launch the interactive web application:
streamlit run app.py

Enter any Amazon food review in the input field.
The app will classify the review as Positive, Negative, or Neutral based on sentiment analysis.

Running Sentiment Analysis from CLI (Command Line Interface)
If you prefer to process reviews directly, run:
python reviews.py
This will analyze predefined reviews in the script and output sentiment classifications.

2️⃣ Performing Exploratory Data Analysis (EDA)
To visualize sentiment distributions and trends:
python eda.py
This script will generate graphs and insights using Matplotlib and Seaborn, helping to understand the dataset better.

3️⃣ Testing the Implementation
To check whether the scripts are updated and working correctly:
python test.py


Example Usage
🔹 Example Review Input:
"The chocolate was amazing! It had the perfect texture and flavor."

🔹 Predicted Sentiment Output:
✅ Positive

🔹 Another Review Input:
"The chips were too salty and stale. Definitely not buying again."

🔹 Predicted Sentiment Output:
❌ Negative




