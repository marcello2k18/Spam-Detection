from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model as keras_load_model  # Renamed to avoid conflict
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Load the trained model
def custom_load_model():
    model = keras_load_model('/Users/roy/Desktop/SMS Spam - Detection/spam_model.h5')  # Correct path to your model
    return model

# Load the tokenizer globally (this makes it accessible in preprocess_message function)
def load_tokenizer():
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Function to preprocess the input message
def preprocess_text(input_text, tokenizer):
    # Clean and preprocess the input text
    text = re.sub("[^a-zA-Z]", " ", input_text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert text to lowercase
    text = text.split()  # Split text into words
    text = [stemmer.stem(word) for word in text if word not in stopwords.words("english")]  # Stem the words
    text = " ".join(text)  # Join the words back into a single string
    
    # Convert the processed text into a sequence for the model
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=47)  # Use the same maxlen as in training
    
    return padded_sequence

# Function to predict if the text is spam or not
def predict_spam(input_text, tokenizer, model):
    # Preprocess the new text
    processed_text = preprocess_text(input_text, tokenizer)
    
    # Predict the probability of the message being spam
    prediction_prob = model.predict(processed_text)[0][0]  # Get the prediction probability
    
    # Return the result based on the threshold
    if prediction_prob > 0.5:
        return "Not Spam"
    else:
        return "Spam"
