import nltk
from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

app = Flask(__name__)

load_vectorizer = pickle.load(open('trained_vectorizer.sav', 'rb'))
load_model = pickle.load(open('trained_model.sav', 'rb'))

port_stem = PorterStemmer()
nltk_stopwords = stopwords.words('english')

def stemming(content):
    stemmed_content= re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content= stemmed_content.lower()
    stemmed_content= stemmed_content.split()
    stemmed_content= [port_stem.stem(word)  for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content= ' '.join(stemmed_content)
    return stemmed_content

@app.route('/', methods=['GET'])
def home():
    return '''
    <h1>This a project using NLP to predict whether a tweet is positive or negative.</h1>
    <p>Send a POST request to /predict with a json payload of the form {"tweet": "Some tweet"}</p>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'tweet' not in data:
        return jsonify({'error': 'No tweet provided.'}), 400
    
    tweet = data['tweet']

    stemmed_tweet = stemming(tweet)
    X_test = load_vectorizer.transform([stemmed_tweet])

    prediction = load_model.predict(X_test)

    result = 'Positive' if prediction[0] == 1 else 'Negative'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
