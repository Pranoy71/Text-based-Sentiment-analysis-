from flask import Flask, render_template, request
import re
import pickle
import numpy as np
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load the sentiment model
sentiment_model = tf.keras.models.load_model("mental_health_classification_model_v4_10000_feature.keras")

# Load the CountVectorizer
with open('CountVectorizer_v4.pkl', 'rb') as file:
    content = file.read()
    cv2 = pickle.loads(content)

# Load the LabelEncoder
with open('encoder_v4.pkl', 'rb') as file:
    content = file.read()
    label_encoder2 = pickle.loads(content)

# Initialize the PorterStemmer
ps = PorterStemmer()

# Preprocessing function
def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line)  # Leave only characters from a to z
    review = review.lower()  # Lower the text
    review = review.split()  # Turn string into list of words
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]  # Delete stop words like I, and, OR
    return " ".join(review)  # Turn list into sentences

# Sentiment detection function
def sentiment_detection(text):
    text = preprocess(text)
    array = cv2.transform([text]).toarray()
    pred = sentiment_model.predict(array)
    a = np.argmax(pred, axis=1)
    sentiment = label_encoder2.inverse_transform(a)[0]
    return sentiment

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = sentiment_detection(text)
        return render_template('index.html', text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

