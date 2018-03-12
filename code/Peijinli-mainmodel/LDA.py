# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:41:28 2018

@author: Peiji
"""

import time
import _pickle as pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pylab
import re
import scipy as sp
import seaborn
import gensim
import sklearn
import os
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

from gensim import corpora, models
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

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

from nltk.corpus import words
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

#define stopwords
#!different with nltk!
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
        texts = [ps.stem(word) for word in review.lower().split() if word not in stoplist]
        #texts = [word for word in review.lower().split() if (word not in stoplist and word in words.words())]
        try:
            clean_data_set.append(' '.join(texts))
        except:
            pass
    return clean_data_set

path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'
os.chdir(path)
from sklearn.cross_validation import train_test_split
# train_test_split function split train and test dataset

file = 'train_data.csv'
data = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
restaurantsDF = subset_givenlength(data,0,10000,150000)
restaurantsDF.index = range(restaurantsDF.shape[0])
restaurantsDF.text= process_reviews(restaurantsDF.text)

#tfidf_vectorizer = TfidfVectorizer()
#textTFIDF = tfidf_vectorizer.fit_transform(restaurantsDF.text)
#textTFIDF_feature_names = tfidf_vectorizer.get_feature_names()

tf_vectorizer = CountVectorizer()
textTF = tf_vectorizer.fit_transform(restaurantsDF.text)
textTF_feature_names = tf_vectorizer.get_feature_names()

train, test = train_test_split(restaurantsDF, test_size=0.2)
train_textTF = textTF[train.index,]
test_textTF = textTF[test.index,]

# LDA can only use raw term counts!!!! for LDA because it is a probabilistic graphical model

path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2\\topic'
os.chdir(path)

no_topics = 10
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(textTF)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 1000
display_topics(lda, textTF_feature_names , no_top_words)

model = lda
for topic_idx, topic in enumerate(model.components_):
    print ("Topic %d" % (topic_idx))
    text = pd.DataFrame([textTF_feature_names [i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    text.to_csv(('').join([str(topic_idx+1),'.csv']))
#X_train = tf_vectorizer.fit_transform(train.text)
#doc_topic_dist_unnormalized_train = np.matrix(lda.transform(X_train))
#doc_topic_dist_train = doc_topic_dist_unnormalized_train/doc_topic_dist_unnormalized_train.sum(axis=1)
#doc_topic_dist_train.argmax(axis=1)

#get top rank
#a = doc_topic_dist_train.argmax(axis=1)[0:50,]
#rank_train=[int(a[i]) for i in range(len(a))]

doc_topic_dist_unnormalized_test = np.matrix(lda.transform(textTF))

# normalize the distribution for test (only needed if you want to work with the probabilities)
doc_topic_dist_test = doc_topic_dist_unnormalized_test/doc_topic_dist_unnormalized_test.sum(axis=1)
doc_topic_dist_test.argmax(axis=1)

#5 star 
#matrix([[0.26240846, 0.00250069, 0.2729901 , 0.0025    , 0.3110452 ,
#         0.0025    , 0.05385163, 0.0872033 , 0.00250059, 0.00250002]])
#matrix([[0.00714586, 0.00714842, 0.00714286, 0.00714286, 0.78585883,
#         0.00714319, 0.15698787, 0.00714344, 0.00714311, 0.00714357]])
#1 star
#matrix([[0.00769281, 0.00769534, 0.00769231, 0.00769648, 0.27169495,
#         0.00769231, 0.00769231, 0.66675824, 0.00769295, 0.00769231]])

#1 star
#matrix([[0.00178588, 0.49806112, 0.00178576, 0.00178614, 0.48765204,
#        0.00178572, 0.00178584, 0.00178599, 0.00178579, 0.00178572]])



restaurantsDF.stars[149996]
doc_topic_dist_test[149996]

X_other = train[[ 'longitude', 'latitude']].copy(deep =True)
X_other.index = range(X_other.shape[0])
X = pd.DataFrame(doc_topic_dist_train[:,0:(no_topics-1)])
re_X = pd.concat([X, X_other], axis=1)

re_X
X2 = pd.DataFrame(doc_topic_dist_test[:,0:(no_topics-1)])
y = train.stars
y2 = test.stars

regr = linear_model.LinearRegression()
regr.fit(X,y)
y_pred = regr.predict(X)

#The coefficients
print('Coefficients: \n', regr.coef_)
#The mean squared error
print("Mean squared error: %.2f")
% mean_squared_error(y, y_pred))

y2_pred = regr.predict(X2)
mean_squared_error(y2, y2_pred)
####dicision tree#########

