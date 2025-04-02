import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'review': [
        # Positive Reviews
        "I absolutely loved this movie! The acting was fantastic.",
        "Amazing film! The plot was engaging and full of surprises.",
        "Fantastic experience, I’d watch it again!",
        "An absolute masterpiece! The cinematography was brilliant.",
        "This movie had a great storyline and excellent direction!",
        "The actors did a phenomenal job, and the visuals were stunning.",
        "I enjoyed every moment of it! Highly recommend to everyone.",
        "A heartwarming movie with a fantastic script and execution.",
        "This was one of the best films I’ve seen all year!",
        "A true work of art, filled with emotions and deep meaning.",
        "Loved the characters and how well-developed the story was.",
        "This film had me hooked from start to finish!",
        "An emotional rollercoaster! One of the finest in recent years.",
        "Great performances and an unforgettable soundtrack.",
        "A beautiful story with top-notch cinematography.",

        # Negative Reviews
        "This was a terrible movie. Waste of time!",
        "I hated this movie. The story was so boring and predictable.",
        "Worst film ever. The dialogues were awful.",
        "Not a good movie. It lacked depth and creativity.",
        "The plot was weak and the characters were unrealistic.",
        "I regret watching this movie, it was incredibly dull.",
        "Nothing made sense! This was a complete disaster.",
        "The direction was awful, and the script was even worse.",
        "An absolute waste of time. I nearly fell asleep watching it.",
        "Terrible storytelling, poor execution, and lackluster acting.",
        "I couldn't relate to any of the characters. Badly written.",
        "Horrible pacing and poor character development.",
        "One of the worst movies I've ever seen. Do not recommend!",
        "Everything about this movie was bad – from acting to story.",
        "This was a complete disappointment, don't waste your time.",
        "This was a waste of time. I walked out halfway through."
    ],
    'sentiment': [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 15 Positive
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # 15 Negative
    ]
}

df = pd.DataFrame(data)


#print(df.head())

X = np.array(df['review'])
y = np.array(df['sentiment'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform on training data (EX: like doing the .fit and the .transform combined is fit_transform)
X_train_tfidf = vectorizer.fit_transform(X_train)
# fit_transform - Learns from the data (calculates statistics) and applies the transformation
X_test_tfidf = vectorizer.transform(X_test)
# transform - Applies the previously learned transformation without learning again

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
#plt.show()

new_reviews = [
    "This movie was an absolute delight, I loved every second!",
    "The worst film I’ve seen in years. Don’t waste your time!",
    "It was just okay. Not great, not terrible."
]

new_reviews_tfidf = vectorizer.transform(new_reviews)
predictions = model.predict(new_reviews_tfidf)

# Print Results
for review, pred in zip(new_reviews, predictions):
    sentiment = "Positive 😊" if pred == 1 else "Negative 😡"
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")

