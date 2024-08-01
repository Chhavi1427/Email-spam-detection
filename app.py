import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    """
    Transform the input text by:
    1. Converting to lowercase
    2. Tokenizing the text
    3. Removing non-alphanumeric characters
    4. Removing stopwords and punctuation
    5. Stemming each word
    """
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]

    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    # Apply stemming
    text = [ps.stem(word) for word in text]

    return " ".join(text)

def load_model_and_vectorizer():
    """
    Load the TF-IDF vectorizer and model from pickle files.
    """
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model .pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

def predict_spam(tfidf, model, input_sms):
    """
    Predict whether the input SMS is spam or not using the loaded model and vectorizer.
    """
    try:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        return result
    except NotFittedError:
        st.error("The TF-IDF vectorizer is not fitted. Please check the model and vectorizer.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.title('Email Spam Classifier')

    # Load the model and vectorizer
    tfidf, model = load_model_and_vectorizer()

    input_sms = st.text_area('Enter the Message')

    if st.button('Predict'):
        result = predict_spam(tfidf, model, input_sms)

        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")

if __name__ == "__main__":
    main()
