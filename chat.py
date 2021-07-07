import random
import json

import numpy as np
from nltk.tokenize import word_tokenize
import torch
from model import NeuralNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def tokenize(sentence):
    return word_tokenize(sentence)

def bag_of_words(tokenized_sentence, words):
    sentence_words = [word for word in tokenized_sentence]

    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag 

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
vocabulary = data['vocabulary']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, vocabulary)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

 
    for intent in intents['intents']:
         if tag == intent["tag"]:
            return random.choice(intent['responses'])
    
    
    