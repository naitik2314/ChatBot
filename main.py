import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        word, labels, training, output = pickle.load(f)
except:
    word = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            word.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    word = [stemmer.stem(w.lower()) for w in word if w != "?"]
    word = sorted(list(set(word)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in word:
            if w in wrds:
                bag.append(1) #We are doing one hot encoding, we are checking, if the word exists, then we would add 1 in the list named bag we created.
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((word, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=10000, batch_size=10, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, word):
    bag = [0 for _ in range(len(word))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(word):
            if w == se:
                bag[i] = 1
    
    return numpy.array(bag)

def chat():
    print("Start Talking with the bot!(type Quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(inp, word)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        print(random.choice(responses))

chat()