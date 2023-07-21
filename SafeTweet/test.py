import sklearn
import pickle
import numpy as np
import gradio as gr
from nltk import *
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import re
import emoji
import string
from nltk.corpus import stopwords


vect = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(words) for words in tokenized])
  
#stem the words
def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

#filter special characters
def filter_special_char(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)
  
#remove extra spaces in the text
def remove_extra_spaces(text):
    return re.sub("\s\s+" , " ", text)
  
  
#clean hashtags
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))
    updated_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet))
    return updated_tweet2
  
  
def remove_emoji(text):
  return emoji.replace_emoji(text, "")


#further cleaning of the text
stop_words = set(stopwords.words('english'))
def strip_all_entities(text): 
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:\@|https?\://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    text = re.sub(r'(.)1+', r'1', text)
    text = re.sub('[0-9]+', '', text)
    stopchars= string.punctuation
    table = str.maketrans('', '', stopchars)
    text = text.translate(table)
    text = [word for word in text.split() if word not in stop_words]
    text = ' '.join(text)
    return text
  
#removing contractions
def remove_contractions(text):
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    return text

#final function to pre-process text by using all the functions
def preprocess(text: str)->str:
    text = remove_emoji(text)
    text = remove_contractions(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = filter_special_char(text)
    text = remove_extra_spaces(text)
    text = stemmer(text)
    text = lemmatize(text)
    return text


def conversion(user_input: str)->str:
  user_input=[preprocess(user_input)]
  
  words = vect.transform(user_input)
  result = model.predict(words)
  
  if result == np.array([1]):
    return("Our model predicts the tweet to be RELIGION based cyberbullying")
  elif result == np.array([2]):
      return("Our model predicts the tweet to be AGE based cyberbullying")
  elif result == np.array([3]):
      return("Our model predicts the tweet to be ETHNICITY based cyberbullying")
  elif result == np.array([4]):
      return("Our model predicts the tweet to be GENDER based cyberbullying")
  elif result == np.array([5]):
      return("Our model predicts the tweet to be OTHER CYBERBULLYING TYPE")
  elif result == np.array([6]):
      return("Our model predicts the tweet to be NOT CYBERBULLYING")
      
      
demo=gr.Interface(fn=conversion,inputs=gr.Textbox(
        label="Input", placeholder="Type your cyberbullying tweet"
    ), outputs=gr.Textbox(label="Prediction"), 
                          title="Safe Tweet")
    
demo.launch(server_name="0.0.0.0")



