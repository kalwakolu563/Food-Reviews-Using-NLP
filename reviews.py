import re
import nltk
import pandas as pd
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
# import dask.dataframe as dd
import matplotlib.pyplot as plt
import seaborn as sns


# For text analysis
from wordcloud import WordCloud, STOPWORDS

# load the dataset
df = pd.read_csv("Reviews.csv")
print(df.columns)


# Drop columns that aren't needed
df = df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator'], axis=1)

# Drop rows with missing values in essential columns
df = df.dropna(subset=['Score', 'Summary', 'Text'])

# Check the shape after cleaning
print(df.shape)


# Data Pre-Processing

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv('Reviews.csv')
df = df[['Score', 'Text']].dropna()

# Create sentiment labels
def convert_score_to_sentiment(score):
    if score in [1, 2]:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'

df['Sentiment'] = df['Score'].apply(convert_score_to_sentiment)

# Text cleaning function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

df['Cleaned_Text'] = df['Text'].apply(clean_text)



sia = SentimentIntensityAnalyzer()

# Apply VADER scores
df['VADER_Score'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Categorize scores
def vader_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['VADER_Sentiment'] = df['VADER_Score'].apply(vader_sentiment)



# Convert text to numeric features
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df['Cleaned_Text']).toarray()
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))









