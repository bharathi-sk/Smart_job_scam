import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the data
# You can change this to 'Test data/test fake  job post.csv' if you prefer
csv_path = 'upload.csv'
df = pd.read_csv(csv_path)

# Select only the features present in your form
features = [
    'Telecommuting',
    'Has_company_logo',
    'Has_questions',
    'Employment_type',
    'Required_experience',
    'Required_education',
    'Function'
]

X = df[features].astype(str)
y = df['Fraudulent']

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Train the model
clf = DecisionTreeClassifier()
clf.fit(X_encoded, y)

# Save the model and encoder
with open('random.pickle', 'wb') as f:
    pickle.dump(clf, f)
with open('encoder.pickle', 'wb') as f:
    pickle.dump(encoder, f)

print("Model and encoder trained and saved using only form features.")
