import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import nltk
from nltk.corpus import stopwords


# Download stopwords
nltk.download("stopwords")


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

data = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)

print("First 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns)


# --------------------------------------------------
# 2. Keep only required columns
# --------------------------------------------------

# The common spam.csv dataset has:
# v1 = label
# v2 = message

data = data.iloc[:, :2]

data.columns = ["label", "message"]

print("\nAfter selecting required columns:")
print(data.head())


# --------------------------------------------------
# 3. Convert labels into numbers
# --------------------------------------------------

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})


# Remove rows where label could not be converted
data = data.dropna(subset=["label", "message"])


# --------------------------------------------------
# 4. Text preprocessing
# --------------------------------------------------

stop_words = set(stopwords.words("english"))


def clean_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Split into words
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


data["message"] = data["message"].apply(clean_text)


# --------------------------------------------------
# 5. Separate features and labels
# --------------------------------------------------

X = data["message"]
y = data["label"]


# --------------------------------------------------
# 6. Split dataset into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 7. Convert text into numerical features using TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# 8. Train Naive Bayes model
# --------------------------------------------------

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)


# --------------------------------------------------
# 9. Make predictions
# --------------------------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------------------------
# 10. Evaluate the model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)


print("\nModel Performance")
print("--------------------")

print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1 Score :", f1)


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Ham", "Spam"]
    )
)


# --------------------------------------------------
# 11. Test the model with new messages
# --------------------------------------------------

messages = [
    "Congratulations! You have won a free prize. Click now!",
    "Hey, are we meeting tomorrow?",
    "URGENT! You have won 1 crore rupees. Call now!",
    "Can you send me the notes?"
]


# Clean the new messages
clean_messages = [
    clean_text(message)
    for message in messages
]


# Convert messages to TF-IDF
message_tfidf = vectorizer.transform(clean_messages)


# Predict
predictions = model.predict(message_tfidf)


# --------------------------------------------------
# 12. Display predictions
# --------------------------------------------------

print("\nNew Message Predictions")
print("------------------------")

for message, prediction in zip(messages, predictions):

    if prediction == 1:
        result = "SPAM"
    else:
        result = "HAM"

    print(f"\nMessage: {message}")
    print(f"Prediction: {result}")