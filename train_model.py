import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the data
df = pd.read_csv('upload.csv')

# Features and target
X = df.drop(['Id', 'Fraudulent'], axis=1)
y = df['Fraudulent']

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X.astype(str))

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)

# Save the model and encoder
with open('random.pickle', 'wb') as f:
    pickle.dump(clf, f)
with open('encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

print("Model and encoder trained and saved as random.pickle and encoder.pickle")