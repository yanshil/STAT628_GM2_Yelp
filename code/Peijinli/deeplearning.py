# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 11:39:44 2018

@author: Peiji
"""
import os
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.engine.topology import Input
from keras.layers import Embedding, Dropout
from keras.layers import Dense
from keras.layers.pooling import GlobalMaxPooling1D
from keras.models import Model

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

from keras.layers import Dense, Flatten, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

import time
import datetime
import os
import _pickle as pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pylab
import re
import scipy as sp
import sklearn
from collections import Counter
from sklearn import svm
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from gensim import corpora, models
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn import datasets, linear_model


from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.decomposition import NMF, LatentDirichletAllocation
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

stoplist ={'ll', 'are', 'theirs', 'up', 'do', 'have', 'who', 'few', 'needn', 'yourselves', 'has', 
           'under', 'ain', 'the', 'should', 'y', 'might', "must", 'was', 'had', 'she', 'is', 'through',
           'himself', 'their', 'ours', 'm', 'and', 'am', 'against', 'his', 'from', 'mustn', 'off',  'her',
           "will", 'myself', 'as', "did", "is", 'themselves', 'o', 'of', 'them', 'does', 'i', 'a', 'by', 
           't', 'had', 'it', 'after', "should've", 'was', 'did', 'my', 'into', 'they', 'such', 'but', 'if', 
           'hers', 'with', 'your', 'than', "had", 'did', "could", "she's",
           "might", 'has', 'each', 'these', 'our', 'will', 'those', 'can', 'he', 'over', 'could', 
           'having', 'below', 'between', 'own', 'until', 'about', 'all', 'being', 'why', 'should', 
           'most', 're', 'we', 'doing', 'at', 'because', 's', 'does', 'now', 'other', 'down', 
           'ourselves', 'so', 'you', 'were', 'while', 'to', 'here', 'me', "you've", 'its', 'herself', 
           'further', 'too', 'isn', "you're", 'were', "was", 'some', 'in', 'been', "it's", 'or', 
           'are', 'nor', "have", 'same', 'before', 'won', 'when', 'more', 'this', 'on', 'only', 'd', "does",
           'both', 'once', 'haven', 'during', "don't", 'very', 'yourself', 'be', 'yours', 'where', 'him', 'what', 
           "you'll", "would", 'that', 'how', 'ma', 'then', "need", "should", 'there', "that'll", 've',
           'an', 'out', 'again', 'itself', 'which', 'wouldn', 'any', 'whom', 'above', "you'd", 'just', 'for', 
           "has", "were"}

#extract abbreviation
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"shan't", "refuse to", phrase)
    phrase = re.sub(r"shan", "happy to", phrase)
    phrase = re.sub(r":\(", "bad", phrase)
    phrase = re.sub(r":\)", "good", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

#given lenth of comment ,random draw sample size
def subset_givenlength(data,minReviewLen,maxReviewLen,subsize):
    t= data.sample(n=subsize, frac=None, replace=False, weights=None, random_state=None, axis=0)
    print("Number of rows selected:",len(t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]))
    subset = t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]
    return subset

def process_reviews(dirty_data_set):
    clean_data_set = []
    for review in dirty_data_set:
        review = decontracted(review)
        # Remove punctuations        
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
        texts = [word for word in review.lower().split() if word not in stoplist]
        try:
            clean_data_set.append(' '.join(texts))
        except:
            pass
    return clean_data_set

path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'

os.chdir(path)
data1 = pd.read_csv('train_data.csv',sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
data2 = pd.read_csv('testval_data.csv',sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')

train, test = data1[0:450000],data1[450000:600000]
texts = process_reviews(data1[0:600000].text)

x_train = train.text
x_test = test.text  

##################################### LSTM ##################################
###################### PRETRAIN
# load the dataset but only keep the top 5000 words, zero the rest
top_words = 50
encoded_docs = [one_hot(d, top_words) for d in texts]
a = [len(x) for x in encoded_docs]
print(max(a))
max_review_length = 800
padded_docs = pad_sequences(encoded_docs, maxlen=max_review_length, padding='post',dtype='float32')
print(padded_docs.shape)
print(len(encoded_docs))

X_train = padded_docs[0:2000,]
X_test = padded_docs[450000:600000,]
######################## encode class values as integers
Y = data1[0:600000].stars
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y.shape)
y_train = dummy_y[0:2000,]
y_test = dummy_y[450000:600000,]

###################### MODEL####################################################
embedding_vecor_length = 128
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(5, activation='softsign'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #optimizer='rmsprop'
#loss="mse"
print(model.summary())
model.fit(X_train, y_train, epochs=30, batch_size=1000)#2000 vs 100
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))





#################### xun huan RNN###########
X_train = np.reshape(X_train, (X_train.shape[0], max_review_length, 1))
X_train = X_train / float(len(encoded_docs))

batch_size = 1
model = Sequential()
model.add(LSTM(32, input_shape=(X_train.shape[1], 1)))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, batch_size=batch_size, verbose=2)
# summarize performance of the model
scores = model.evaluate(X_train, y_train, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

