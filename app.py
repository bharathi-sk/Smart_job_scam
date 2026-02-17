import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename
import pickle

app = Flask(__name__) #Initialize the flask App


model = pickle.load( open('random.pickle', 'rb') )
vecs = pickle.load( open('vectorizers.pickle', 'rb') )
classifiers = pickle.load( open('classifiers.pickle', 'rb') )
encoder = pickle.load(open('encoder.pickle', 'rb'))


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')
   

@app.route('/login')
def login():
    return render_template('login.html')
@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)
        
@app.route('/fake_prediction')
def fake_prediction():
    return render_template('fake_prediction.html')



@app.route('/predict', methods=['POST'])
def predict():
    # Load encoder
    with open('encoder.pickle', 'rb') as f:
        encoder = pickle.load(f)

    # Collect input features from form
    input_dict = {key: value for key, value in request.form.items()}

    # Debug: print raw input from form
    print("==== RAW FORM INPUT ====")
    print(input_dict)

    # Convert to DataFrame (single row)
    input_df = pd.DataFrame([input_dict])

    # Ensure all values are strings (encoder expects strings)
    input_df = input_df.astype(str)

    # Add missing columns with default empty string values
    expected_cols = encoder.feature_names_in_
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = ""

    # Reorder columns to match encoder
    input_df = input_df[expected_cols]

    # Debug: print dataframe before encoding
    print("==== DATAFRAME BEFORE ENCODING ====")
    print(input_df)

    # Transform using encoder
    final_features = encoder.transform(input_df)

    # Debug: check encoded shape
    print("==== ENCODED FEATURE SHAPE ====")
    print(final_features.shape)

        # Make prediction
    y_pred = model.predict(final_features)
    y_prob = model.predict_proba(final_features)

    print("==== MODEL RAW PREDICTION ====", y_pred)
    print("==== MODEL PROBABILITIES ====", y_prob)

    if y_pred[0] == 1:
        prediction_texts = "Fake Job Post"
    else:
        prediction_texts = "Legit Job Post"

    return render_template(
        'fake_prediction.html',
        prediction_texts=prediction_texts
    )


@app.route('/text_prediction')
def text_prediction():
     return render_template("text_prediction.html")

 

@app.route('/job')
def job():	
    abc = request.args.get('news')	
    input_data = [abc.rstrip()]
    # transforming input
    tfidf_test = vecs.transform(input_data)
    # predicting the input
    y_preds = classifiers.predict(tfidf_test)
    if y_preds[0] == 1:
        labels = "Fake Job Post"
    elif y_preds[0] == 0:
        labels = "Legit Job Post"
    else:
        labels = f"Prediction: {y_preds[0]}"
    return render_template('text_prediction.html', prediction_text=labels)
    
    
if __name__ == "__main__":
    app.run(debug=True)
