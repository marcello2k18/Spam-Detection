from flask import Flask, render_template, request
import pickle
import numpy as np
from model import custom_load_model, load_tokenizer, predict_spam

app = Flask(__name__)

# Load the trained model and tokenizer
model = custom_load_model()  # Use custom_load_model instead of load_model
tokenizer = load_tokenizer()  # Use the load_tokenizer function

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    if request.method == "POST":
        message = request.form["message"]
        
        # Predict if the message is spam or not using the model
        prediction_result = predict_spam(message, tokenizer, model)  # Pass the message to predict_spam

    return render_template("index.html", prediction_result=prediction_result)

if __name__ == "__main__":
    app.run(debug=True)
