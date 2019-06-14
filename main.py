import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow as tf
import random
import json

stemmer = LancasterStemmer()
with open("intents.json")as file:
    data = json.load(file)

print(data)

words = []
labels = []
docs= []

for intent in data["intents"]:
    for patterns in intent["patterns"]:
        wrds = nltk.word_tokenize(patterns)
