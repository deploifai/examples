from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import gradio as gr
import numpy as np
import pickle

# import the model
model = pickle.load(open("model.pkl", "rb"))


# This function uses a pre-trained machine learning model to detect
# whether a user input string is a spam SMS or not and returns a corresponding message.
def results(user_input: str):
    user_input = pd.Series(user_input)
    if model.predict(user_input) == np.array([1]):
        return "Spam SMS detected"
    else:
        return "The SMS is NOT spam"


# gradio interface for the model
demo = gr.Interface(
    fn=results,
    inputs=gr.Textbox(label="Input", placeholder="Type your SMS Text"),
    outputs=gr.Textbox(label="Prediction"),
    title="SPAM SMS CLASSIFIER",
)

demo.launch(server_name="0.0.0.0")
