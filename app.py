from fastapi import FastAPI
import asyncio
import gradio as gr
import re
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
import pickle

# Function to remove URLs from text
def remove_urls(text):
    return re.sub(r'http[s]?://\S+', '', text)

# Function to remove punctuations from text
def remove_punctuation(text):
    regular_punct = string.punctuation
    return str(re.sub(r'['+regular_punct+']', '', str(text)))

# Function to convert the text into lower case
def lower_case(text):
    return text.lower()

# Function to lemmatize text
def lemmatize(text):
    wordnet_lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(text)
    lemma_txt = ''
    for w in tokens:
        lemma_txt = lemma_txt + wordnet_lemmatizer.lemmatize(w) + ' '

    return lemma_txt

def load_model():
    # Define the file path where the trained model is saved
    model_file_path = "logistic_regression_model.pkl"

    # Load the saved Logistic Regression model from the file
    with open(model_file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model

def load_tfidf():
    # Define the file path where the TF-IDF vectorizer is saved
    vectorizer_file_path = "tfidf_vectorizer.pkl"

    # Load the saved TF-IDF vectorizer from the file
    with open(vectorizer_file_path, 'rb') as file:
        loaded_vectorizer = pickle.load(file)

    return loaded_vectorizer

def preprocess(input_text):
    # Preprocess the input text
    input_text = remove_urls(input_text)
    input_text = remove_punctuation(input_text)
    input_text = lower_case(input_text)
    input_text = lemmatize(input_text)

    # Apply TF-IDF vectorization
    input_text = [input_text]
    tfidf = load_tfidf()
    input_text = tfidf.transform(input_text)

    return input_text

app = FastAPI()

@app.get('/')
async def welcome():
    return "Welcome to our Sentiment Analysis API"

@app.post('/predict_sentiment')
async def predict_sentiment(input_text):
    loaded_model = load_model()
    predicted_sentiment = loaded_model.predict(preprocess(input_text))
    if predicted_sentiment == 0:
        sentiment = "Sentiment: Negative"
    else:
        sentiment = "Sentiment: Positive"
    return sentiment

async def predict(input):
    sentiment = await predict_sentiment(input)
    return sentiment

# Create Gradio interface 
iface = gr.Interface(fn=predict, inputs="text", outputs="text", title="Movie Review Sentiment Analysis API", description="Enter a review to know its sentiment...")
iface.launch(share=True)

asyncio.run(predict())