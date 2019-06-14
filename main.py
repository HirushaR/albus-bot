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
docs_x= []
docs_y

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
words = [stammer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels= sorted(labels)

traning = []
output= []

out_empty = [ 0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds =[stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    traning.append(bag)
    output.append(output_row)

traning = numpy.array(traning)
output = np.array(output)
