import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the training data
train_df = pd.read_csv('Test data/test text_prediction.csv')

# Prepare text and target
texts = train_df['description'].fillna("")
y = train_df['fraudulent'].fillna("")

# Fit the vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train the classifier
clf = KNeighborsClassifier()
clf.fit(X, y)

# Save the fitted vectorizer
with open('vectorizers.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the trained classifier
with open('classifiers.pickle', 'wb') as f:
    pickle.dump(clf, f)

print("Fitted TfidfVectorizer and KNeighborsClassifier saved as vectorizers.pickle and classifiers.pickle")
