# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:05:34 2018

@author: Peiji
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:49:15 2018

@author: Peiji
"""

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
        try:
            clean_data_set.append(' '.join(texts))
        except:
            pass
    return clean_data_set

def getsp_category(catlist):
    yelp_categories_tidy = [re.sub("\'", '', x.strip("[]")).split(',') for x in data.categories]
    categories_counter = Counter()
    for x in yelp_categories_tidy:
        categories_counter.update(x)
    categories_dict = dict(categories_counter)
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer()
    categories_sp = v.fit_transform(Counter(f) for f in yelp_categories_tidy)
    return [categories_sp , list(categories_counter.keys())]
    

path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'
os.chdir(path)
from sklearn.cross_validation import train_test_split
# train_test_split function split train and test dataset

file = 'train_data.csv'
data = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
restaurantsDF = subset_givenlength(data,0,10000,30000)#data
restaurantsDF.textlen = pd.Series([len(i) for i in restaurantsDF.text])

restaurantsDF.index = range(restaurantsDF.shape[0])
restaurantsDF.text= process_reviews(restaurantsDF.text)

nofeature = 1000000

tf_vectorizer = CountVectorizer(max_features= nofeature)
textTF = tf_vectorizer.fit_transform(restaurantsDF.text)
textTF_feature_names = tf_vectorizer.get_feature_names()

train, test = train_test_split(restaurantsDF, test_size=0.2)
train_textTF = textTF[train.index,]
test_textTF = textTF[test.index,]

############################################
category = getsp_category(restaurantsDF.categories)[0]
train_cat = category[train.index,]
test_cat = category[test.index,]
train_len = restaurantsDF.textlen[train.index]
test_len = restaurantsDF.textlen[test.index]

#weight of class
def getproportion(dataset, labelname):
    weight = {}
    total = sum(dataset.groupby(labelname).size())
    for i in dataset.groupby(labelname).size().index:
        weight[i] = train.groupby(labelname).size().get(i)/total
    return weight


from scipy.sparse import hstack
from sklearn import svm


########################################### TF################################################

X_train2 = hstack((train_textTF,np.array(train['longitude'])[:,None],
                  np.array(train['latitude'])[:,None],train_cat,np.array(train_len)[:,None]))

X_test2 = hstack((test_textTF,np.array(test['longitude'])[:,None],
                  np.array(test['latitude'])[:,None],test_cat,np.array(test_len)[:,None]))

from sklearn.naive_bayes import GaussianNB
weight = getproportion(train,'stars')
clf = GaussianNB()
clf.fit(X_train2.toarray(), train.stars)

predY = clf.predict(X_test2.toarray())
mean_squared_error(test.stars, predY)#4.02






########################################### predict test ####################################
file2 = 'testval_data.csv'
data2 = pd.read_csv(file2,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
data2.text= process_reviews(data2.text)
frames = [data.text, data2.text]
frames2 = [data.categories,data2.categories]
total_text = pd.concat(frames)
total_text.to_csv('totaltext.csv')
total_cat = pd.concat(frames2)
total_cat.to_csv('totalcat.csv')

##################################fasttext######################################################
import fasttext

# Skipgram model
model = fasttext.skipgram('data.txt', 'model')
print model.words # list of words in dictionary

# CBOW model
model = fasttext.cbow('data.txt', 'model')
print model.words # list of words in dictionary
