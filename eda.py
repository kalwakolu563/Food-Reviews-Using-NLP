import re
import time
import pandas as pd
import numpy as np
# import dask.dataframe as dd
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
import seaborn as sns


# For text analysis
from wordcloud import WordCloud, STOPWORDS
# step 1 : Load the data
df = pd.read_csv("Reviews.csv")
print(df.columns)

# Basic Exploratory Data Analysis

# Step 2: Data Cleaning

print(df.info())
df.head()
df.tail()

df.isnull().sum()
print(df.describe())

# Drop columns that aren't needed
df = df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'Time', 'HelpfulnessNumerator', 'HelpfulnessDenominator'], axis=1)

# Drop rows with missing values in essential columns
df = df.dropna(subset=['Score', 'Summary', 'Text'])

# Check the shape after cleaning
print(df.shape)


# Plot the distribution of scores
plt.figure(figsize=(8, 5))
sns.countplot(x='Score', data=df,hue='Score', palette='viridis', legend=False)
plt.title('Distribution of Review Scores')
plt.xlabel('Review Score')
plt.ylabel('Number of Reviews')
plt.show()

time.sleep(2)


# Combine all text for word cloud
text = " ".join(review for review in df.Text)

# Generate a word cloud
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=800, height=400).generate(text)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Common Words in Reviews")
plt.show()

time.sleep(2)


# Add a column for review length
df['Review_Length'] = df['Text'].apply(lambda x: len(x.split()))

# Plot average review length by score
plt.figure(figsize=(8, 5))
sns.barplot(x='Score', y='Review_Length', hue='Score', data=df, palette='magma', legend=False)
plt.title('Average Review Length by Score')
plt.xlabel('Score')
plt.ylabel('Average Number of Words')
plt.show()

time.sleep(2)


# Check for duplicates
duplicates = df.duplicated(subset=['Text']).sum()
print(f"Number of duplicate reviews: {duplicates}")

# Drop duplicate reviews
df = df.drop_duplicates(subset=['Text'])

# Select only numerical columns
numeric_df = df.select_dtypes(include=['number'])

# Plot correlation heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

time.sleep(2)


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


# Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sentiment', data=df)
plt.title('Sentiment Distribution')
plt.show()

# WordCloud for positive reviews
positive_words = ' '.join(df[df['Sentiment'] == 'Positive']['Cleaned_Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Positive Reviews')
plt.show()




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

import matplotlib.pyplot as plt
import seaborn as sns

# Extract individual VADER scores into separate columns
df['Positive'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['pos'])
df['Neutral'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['neu'])
df['Negative'] = df['Cleaned_Text'].apply(lambda x: sia.polarity_scores(x)['neg'])

# Plot the sentiment scores
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Plot for Positive Sentiment
sns.barplot(x=df.index, y=df['Positive'], ax=axs[0], color='green')
axs[0].set_title('Positive Sentiment')
axs[0].set_xlabel('Review Index')
axs[0].set_ylabel('Positive Score')

# Plot for Neutral Sentiment
sns.barplot(x=df.index, y=df['Neutral'], ax=axs[1], color='blue')
axs[1].set_title('Neutral Sentiment')
axs[1].set_xlabel('Review Index')
axs[1].set_ylabel('Neutral Score')

# Plot for Negative Sentiment
sns.barplot(x=df.index, y=df['Negative'], ax=axs[2], color='red')
axs[2].set_title('Negative Sentiment')
axs[2].set_xlabel('Review Index')
axs[2].set_ylabel('Negative Score')

# Layout adjustments
plt.tight_layout()
plt.show()



