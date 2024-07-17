# Sentiment Analysis Web Application

This project is a web application for sentiment analysis, which allows users to input text and receive a sentiment prediction. The application is built using Flask and leverages a deep learning model to perform sentiment analysis on the given text.

## Project Overview

The main components of the project include:

1. **Deep Learning Model**: The core of the sentiment analysis functionality is a CNN-based deep learning model. This model uses 1D Convolutional Neural Network (CNN) layers to extract features from text data. The model is trained with a CountVectorizer that has a maximum feature size of 10,000 and considers n-grams ranging from 1 to 3.

2. **Flask Web Application**: The frontend of the application is built using HTML and CSS to provide a user-friendly interface. Users can enter text into a form, and upon submission, the text is sent to the backend Flask application, which processes the text and returns the sentiment prediction.

## Model Details

- **Architecture**: The model is a CNN-based deep learning model that utilizes 1D CNN layers to extract textual features.
- **Vectorization**: The text is vectorized using a CountVectorizer with a maximum of 10,000 features and n-grams ranging from 1 to 3.
- **Preprocessing**: Text preprocessing includes converting text to lowercase, removing non-alphabetic characters, stemming, and removing stop words.

## Project Structure

- /project folder
- /templates
- index.html
- app.py
- CountVectorizer_v4.pkl # CountVectorizer file
- encoder_v4.pkl # LabelEncoder file
- mental_health_classification_model_v4_10000_feature.keras # Trained model file




## Usage

1. Open the web application in your browser.
2. Enter the text you want to analyze in the provided text area.
3. Click the "Analyze Sentiment" button.
4. The application will display the sentiment prediction for the entered text.


## Conclusion

This project demonstrates the integration of a CNN-based deep learning model with a Flask web application to perform sentiment analysis. The application provides a simple and intuitive interface for users to input text and receive sentiment predictions.

## Acknowledgments

- TensorFlow and Keras for the deep learning framework.
- Flask for the web framework.
- NLTK for text preprocessing tools.


## Developed by Pranoy71 