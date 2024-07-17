# Mental Health Chatbot Project

## Overview

This project focuses on the development of a mental health chatbot using deep learning techniques. The chatbot is designed to engage users in meaningful conversations and provide valuable feedback based on their input. The core of this project lies in a CNN-based deep learning model, which processes text data to understand user sentiment and generate appropriate responses.

## Project Features

- **Deep Learning Model:** The project employs a Convolutional Neural Network (CNN) model with 1D CNN layers to extract features from text data.
- **Text Vectorization:** Utilizes CountVectorizer with a maximum of 10,000 features and n-grams ranging from 1 to 3 for text preprocessing.
- **Flask App:** A Flask web application serves as the interface for user interactions, allowing users to communicate with the chatbot seamlessly.

## Model Details

The deep learning model used in this project is designed to process text data efficiently. The key components of the model are:

- **1D CNN Layers:** These layers are responsible for extracting relevant features from the text data, capturing important patterns and structures.
- **CountVectorizer:** This tool converts the text data into a matrix of token counts, with a maximum feature limit of 10,000 and n-grams ranging from 1 to 3.

## Flask App

The Flask application provides a user-friendly interface for interacting with the chatbot. It handles user inputs, processes them through the deep learning model, and returns appropriate responses. The app is designed to be lightweight and responsive, ensuring a smooth user experience.

## Getting Started

To run the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mental-health-chatbot.git
   cd mental-health-chatbot
