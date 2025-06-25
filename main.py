import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset (Replace with your actual file path)
df = pd.read_csv('/content/SENTI.csv')

# -----------------------------
# **Text Preprocessing Function**
# -----------------------------
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters & numbers
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['Processed_Text'] = df['Text'].astype(str).apply(preprocess_text)

# -----------------------------
# **Sentiment Analysis with TextBlob**
# -----------------------------
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['Sentiment_Polarity'] = df['Processed_Text'].apply(get_sentiment)

# Assign sentiment labels
df['Sentiment_Label'] = df['Sentiment_Polarity'].apply(lambda x: 'positive' if x > 0.1 else ('negative' if x < -0.1 else 'neutral'))
df['Sentiment_Label_Numeric'] = df['Sentiment_Label'].map({'positive': 1, 'negative': 0, 'neutral': 2})

# -----------------------------
# **Vectorization using TF-IDF**
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to prevent overfitting
X = vectorizer.fit_transform(df['Processed_Text'])
y = df['Sentiment_Label_Numeric']

# -----------------------------
# **Train-Test Split**
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# **Model 1: Logistic Regression with GridSearch**
# -----------------------------
log_reg = LogisticRegression(max_iter=1000)
param_grid = {'C': [0.1, 1, 10]}  # Regularization strength tuning
grid_lr = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_lr.fit(X_train, y_train)
best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test)

# -----------------------------
# **Model 2: Naïve Bayes**
# -----------------------------
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# -----------------------------
# **Model Evaluation**
# -----------------------------
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=['positive', 'negative', 'neutral']))

print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(classification_report(y_test, y_pred_nb, target_names=['positive', 'negative', 'neutral']))

# -----------------------------
# **Data Parsing for Time Analysis**
# -----------------------------
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')

# -----------------------------
# **Visualization 1: Sentiment Distribution by Country**
# -----------------------------
def plot_geography_trends(df):
    # Ensure unique country names by sorting and grouping correctly
    country_sentiment = df.groupby(['Country', 'Sentiment_Label']).size().unstack(fill_value=0)

    # Sort countries for better visualization
    country_sentiment = country_sentiment.sort_values(by=country_sentiment.columns.tolist(), ascending=False)

    fig, ax = plt.subplots(figsize=(16, 6))  # Increased figure size
    country_sentiment.plot(kind='bar', stacked=True, colormap='coolwarm', ax=ax)

    ax.set_title('Sentiment Distribution by Country', fontsize=14)
    ax.set_xlabel('Country', fontsize=12)
    ax.set_ylabel('Number of Tweets', fontsize=12)

    # Ensure each country name appears only once
    plt.xticks(range(len(country_sentiment.index)), country_sentiment.index, rotation=90, fontsize=8, ha='right')

    ax.legend(title='Sentiment', loc='upper right')
    plt.tight_layout()
    plt.show()


# -----------------------------
# **Visualization 2: Sentiment Trends Over Time (By Date)**
# -----------------------------
def plot_time_trends(df):
    time_sentiment = df.groupby(['Date', 'Sentiment_Label']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))  # Increased width
    time_sentiment.plot(kind='line', colormap='coolwarm', ax=ax)

    ax.set_title('Sentiment Trends Over Time', fontsize=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Tweets', fontsize=12)

    # Fix x-labels for date readability
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Slant labels for clarity

    plt.legend(title='Sentiment', loc='upper left')
    plt.tight_layout()
    plt.show()

# -----------------------------
# **Visualization 3: Sentiment Trends by Hour**
# -----------------------------
def plot_hourly_trends(df):
    hourly_sentiment = df.groupby(['Hour', 'Sentiment_Label']).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    hourly_sentiment.plot(kind='line', colormap='coolwarm', ax=ax)

    ax.set_title('Sentiment Trends by Hour', fontsize=14)
    ax.set_xlabel('Hour', fontsize=12)
    ax.set_ylabel('Number of Tweets', fontsize=12)

    # Fix x-labels for better hour readability
    plt.xticks(range(0, 24, 2), fontsize=10)  # Show every 2 hours

    plt.legend(title='Sentiment', loc='upper left')
    plt.tight_layout()
    plt.show()

# -----------------------------
# **Plot Results**
# -----------------------------
plot_geography_trends(df)
plot_time_trends(df)
plot_hourly_trends(df)
