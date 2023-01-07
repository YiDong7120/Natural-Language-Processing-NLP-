from flask import Flask, request, render_template
import pickle
import pandas as pd
from textPreprocessing import text_preprocessing

app = Flask(__name__)
model = pickle.load(open('model_lr.pkl', 'rb'))
tf_idf_transformer = pickle.load(open('tf_idf_transformer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = pd.DataFrame({"review" : [request.form.get("movie_review")]})
    processed_text = text_preprocessing(text)
    
    vector=tf_idf_transformer.transform(processed_text)
    vector = vector.toarray()
    prediction = model.predict(vector)
    if(prediction == 1):
        output = 'Positive'
    else :
        output = 'Negative'
    return render_template('index.html', prediction_text='{} review'.format(output))

if __name__ == "__main__":
    app.run(debug=True)