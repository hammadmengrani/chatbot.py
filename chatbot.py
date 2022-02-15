import random
import json
import pickle

import numpy
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open('words.pkl', 'rb'))
words = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot model.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_word (sentence):
    sentence_word = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_word:
        for i , word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_word(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i , r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

        return return_list

print("GO! bot is running")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints , intents)
    print(res)






