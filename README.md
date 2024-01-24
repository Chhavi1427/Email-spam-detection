# Email/SMS Spam Detection


[![Email Spam Detection](https://github.com/Chhavi1427/Email-spam-detection/assets/115630286/fcaa528c-8120-4f3e-a201-cc78650ed0ad)](http://localhost:8501/)


This is a simple Spam Classifier web application built using Streamlit and a pre-trained machine learning model. The model is trained to predict whether a given text message is spam or not. It utilizes natural language processing techniques, including text preprocessing, tokenization, and stemming.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Getting Started](#getting-started)
  - [Running the Application](#running-the-application)
- [Features](#features)
- [Files](#files)
- [Technologies](#technologies)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Prerequisites

The code you provided has some prerequisites in terms of libraries and model/data files. Here are the prerequisites you need to consider:

**1. Python:**
- Ensure you have Python installed on your system. You can download it from [Python's official website](https://www.python.org/).

**2.Libraries:** 

- Install the required libraries using the following commands in your terminal or command prompt:
```bash
pip install streamlit nltk scikit-learn
```
- Additionally, you may need to download NLTK data. Add the following code at the beginning of your script:
``` python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**3. Streamlit:**

Streamlit should be installed using the following command:
```bash
pip install streamlit

```
## Usage

**Getting Started**

1 Clone the repository to your local machine:

git clone https://github.com/Chhavi1427/Email-spam-detection.git

cd Email-spam-detection

2 Install the required dependencies. Make sure you have Python installed: 
```bash  
pip install -r requirements.txt
```

**Running the Application**


3.Run the Streamlit app:

```bash
streamlit run app.py
```
4 Open your web browser and go to http://localhost:8501/ to interact with the Spam Detection.


## Features

- **Text-Preprocessing:** The input text undergoes several preprocessing steps, including lowercase conversion, tokenization, and stemming.

- **Stopword-Removal:** Common English stopwords are removed to focus on important words.

- **Machine Learning Model:** The pre-trained model uses a vectorizer (TF-IDF) and a machine learning model for prediction.
- **Light/Dark Mode Toggle:** The application provides the flexibility to switch between light and dark modes for a personalized viewing experience.

- **Live Previews:** Users can see live previews of their input and the model's prediction instantly.

- **Fullscreen Mode:** The application supports fullscreen mode, allowing users to focus solely on the prediction results.

- **Cross-Platform:** The application is designed to work seamlessly across different platforms.


## Files

- `app.py`: The Streamlit application code.
- `model.pkl`: Pre-trained machine learning model for spam detection.
- `vectorizer.pkl`: Pre-trained TF-IDF vectorizer for text transformation.
- `requirements.txt`: List of required Python packages.
## Technologies

- [Streamlit](https://www.streamlit.io/ "Streamlit Official Website")

- [NLTK](https://www.nltk.org/)



## Contact 

**Project Name:** Email/SMS Spam Classifier

**Author:** [Chhavi Modi ](https://github.com/Chhavi1427)

**Project Link:** https://github.com/Chhavi1427/Email-spam-detection

**Email:** modichavi1427@gmail.com

## Acknowledgements

 - The project utilizes the NLTK library for natural language processing.
