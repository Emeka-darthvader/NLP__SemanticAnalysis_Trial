import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D,Conv1D,LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


import seaborn as sns

from numpy import array
from numpy import asarray
from numpy import zeros

import matplotlib.pyplot as plt




movie_reviews = pd.read_csv("Data/IMDB Dataset.csv")

movie_reviews.isnull().values.any()

movie_reviews.shape

print("reviewing top 5 rows")
print("*"*20)
print(movie_reviews.head())
print("*"*20)

#viewing the distributing of negative and postive sentiments
# sns.countplot(x='sentiment', data=movie_reviews)



# tutorial link https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

X = []
sentences = list(movie_reviews['review'])
for sen in sentences:
    X.append(preprocess_text(sen))

# print(X[3])

# y = movie_reviews['sentiment']

#replacing positives with 1 and negatives with 0
# y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

#using train, test split from sklearn to divide the training sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


#Embedding Layer
#tokenizing text
tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(X_train)

# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)


# # Using length of 100 for lists where a list is a sentence.
vocab_size = len(tokenizer.word_index) + 1
maxlen = 100

# X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


# #Using GloVe to create word embeddings

# embeddings_dictionary = dict()
# glove_file = open('Word Embeddings/glove.6B.100d.txt', encoding="utf8")

# for line in glove_file:
#     records = line.split()
#     word = records[0]
#     vector_dimensions = asarray(records[1:], dtype='float32')
#     embeddings_dictionary [word] = vector_dimensions
# glove_file.close()


instance = X[57]
print(instance)

instance = tokenizer.texts_to_sequences(instance)

flat_list = []
for sublist in instance:
    for item in sublist:
        flat_list.append(item)

flat_list = [flat_list]

instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)


#creatting Embedding matrix with 100 columns

embeddings_dictionary = dict()
glove_file = open('Word Embeddings/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()



embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# #simpleNN
# model = Sequential()
# embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
# model.add(embedding_layer)

# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))


# #Compile model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.load_weights('models/SimpleNN.h5')

# #CNN
model = Sequential()

embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

#Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.load_weights('models/CNN.h5')

# #LSTM
# model  = load_model('models/LSTM.h5')

model.predict(instance)
