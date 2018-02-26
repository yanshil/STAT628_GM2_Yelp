# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 02:16:17 2018

@author: Peiji
"""
import time
import datetime

import _pickle as pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pylab
import re
import scipy as sp
import seaborn

from gensim import corpora, models
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
#from sklearn.lda import LDA
#from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'
os.chdir(path)

file = 'train_data.csv'
data = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
sample = data.sample(n=150000, frac=None, replace=False, weights=None, random_state=None, axis=0)



###Advanced model###############################
stoplist = set(stopwords.words("english"))
numTopics = 15
t = sample
minReviewLen = 50
maxReviewLen = 100
print("Number of rows selected:",len(t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]))
restaurantsDF = t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]


def perform_lda(allReviewsTrain, numTopics):
    corpus = []
    for review in allReviewsTrain:
        # Remove punctuations
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
        texts = [word for word in review.lower().split() if word not in stoplist]
        try:
            corpus.append(texts)
        except:
            pass

    # Build dictionary
    dictionary = corpora.Dictionary(corpus)
    dictionary.save('restaurant_reviews.dict')
        
    # Build vectorized corpus
    corpus_2 = [dictionary.doc2bow(text) for text in corpus]
    #corpora.MmCorpus.serialize('LDA/restaurant_reviews.mm', corpus_2)
    
    lda = models.LdaModel(corpus_2, num_topics=numTopics, id2word=dictionary)
    return lda

def process_reviews(dirty_data_set):
    clean_data_set = []
    for review in dirty_data_set:
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

# Generates a matrix of topic probabilities for each document in matrix
# Returns topic_dist for the input corpus, and all_dist, a running sum of all the corpuses
def generate_topic_dist_matrix(lda, numTopics, corpus, all_dist, star):
    topic_dist = [0] * numTopics
    dictionary = corpora.Dictionary.load("restaurant_reviews.dict")
    for doc in corpus:
        vec = dictionary.doc2bow(doc.lower().split())
        output = lda[vec]
        highest_prob = 0
        highest_topic = 0
        temp = [0] * numTopics    # List to keep track of topic distribution for each document
        for topic in output:
            this_topic, this_prob = topic
            temp[this_topic] = this_prob
            if this_prob > highest_prob:
                highest_prob = this_prob 
                highest_topic = this_topic
        temp.append(star)
        all_dist.append(temp)
        topic_dist[highest_topic] += 1
    return topic_dist, all_dist

starsGroup = restaurantsDF.groupby('stars')

all_1stars_text = starsGroup.get_group(1.0)['text']
all_2stars_text = starsGroup.get_group(2.0)['text']
all_3stars_text = starsGroup.get_group(3.0)['text']
all_4stars_text = starsGroup.get_group(4.0)['text']
all_5stars_text = starsGroup.get_group(5.0)['text']

all_1stars_labels = [1.0]*len(all_1stars_text)
all_2stars_labels = [2.0]*len(all_2stars_text)
all_3stars_labels = [3.0]*len(all_3stars_text)
all_4stars_labels = [4.0]*len(all_4stars_text)
all_5stars_labels = [5.0]*len(all_5stars_text)

from sklearn.cross_validation import train_test_split

all_1stars_text_train, all_1stars_text_test, all_1stars_labels_train, all_1stars_labels_test = train_test_split(all_1stars_text, all_1stars_labels, test_size=0.20)
all_2stars_text_train, all_2stars_text_test, all_2stars_labels_train, all_2stars_labels_test = train_test_split(all_2stars_text, all_2stars_labels, test_size=0.20)
all_3stars_text_train, all_3stars_text_test, all_3stars_labels_train, all_3stars_labels_test = train_test_split(all_3stars_text, all_3stars_labels, test_size=0.20)
all_4stars_text_train, all_4stars_text_test, all_4stars_labels_train, all_4stars_labels_test = train_test_split(all_4stars_text, all_4stars_labels, test_size=0.20)
all_5stars_text_train, all_5stars_text_test, all_5stars_labels_train, all_5stars_labels_test = train_test_split(all_5stars_text, all_5stars_labels, test_size=0.20)

corpus_5stars = process_reviews(all_5stars_text_train)
corpus_4stars = process_reviews(all_4stars_text_train)
corpus_3stars = process_reviews(all_3stars_text_train)
corpus_2stars = process_reviews(all_2stars_text_train)
corpus_1stars = process_reviews(all_1stars_text_train)

print("Number of 5-star reviews after processing: ", len(corpus_5stars))
print("Number of 4-star reviews after processing: ", len(corpus_4stars))
print("Number of 3-star reviews after processing: ", len(corpus_3stars))
print("Number of 2-star reviews after processing: ", len(corpus_2stars))
print("Number of 1-star reviews after processing: ", len(corpus_1stars))

all_5_4_train = np.append(corpus_5stars, corpus_4stars)
all_5_4_3_train = np.append(all_5_4_train, corpus_3stars)
all_5_4_3_2_train = np.append(all_5_4_3_train, corpus_2stars)
all_text_train = np.append(all_5_4_3_2_train, corpus_1stars)

%time lda = perform_lda(all_text_train, numTopics)

topic_dist_list = []
# Keep a separate list to count topics
topic_dist_5stars = []
topic_dist_4stars = []
topic_dist_3stars = []
topic_dist_2stars = []
topic_dist_1stars = []

topic_dist_5stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_5stars, topic_dist_list, 5)
topic_dist_4stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_4stars, topic_dist_list, 4)
topic_dist_3stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_3stars, topic_dist_list, 3)
topic_dist_2stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_2stars, topic_dist_list, 2)
topic_dist_1stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_1stars, topic_dist_list, 1)

cols = []
for i in range(1, numTopics+1):
    cols.append("Topic"+ str(i))
cols.append("Star")
topic_dist_train_1_2_3_4_5_df = pd.DataFrame(topic_dist_list, columns=cols)

# Process the test reviews
corpus_5stars = process_reviews(all_5stars_text_test)
corpus_4stars = process_reviews(all_4stars_text_test)
corpus_3stars = process_reviews(all_3stars_text_test)
corpus_2stars = process_reviews(all_2stars_text_test)
corpus_1stars = process_reviews(all_1stars_text_test)

print ("Number of 5-star reviews after processing: ", len(corpus_5stars))
print ("Number of 4-star reviews after processing: ", len(corpus_4stars))
print ("Number of 3-star reviews after processing: ", len(corpus_3stars))
print ("Number of 2-star reviews after processing: ", len(corpus_2stars))
print ("Number of 1-star reviews after processing: ", len(corpus_1stars))

all_5_4_test = np.append(corpus_5stars, corpus_4stars)
all_5_4_3_test = np.append(all_5_4_test, corpus_3stars)
all_5_4_3_2_test = np.append(all_5_4_3_test, corpus_2stars)
all_text_test = np.append(all_5_4_3_2_test, corpus_1stars)

topic_dist_list = []

# Keep a separate list to count topics
topic_dist_5stars = []
topic_dist_4stars = []
topic_dist_3stars = []
topic_dist_2stars = []
topic_dist_1stars = []


topic_dist_5stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_5stars, topic_dist_list, 5)
topic_dist_4stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_4stars, topic_dist_list, 4)
topic_dist_3stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_3stars, topic_dist_list, 3)
topic_dist_2stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_2stars, topic_dist_list, 2)
topic_dist_1stars, topic_dist_list = generate_topic_dist_matrix(lda, numTopics, corpus_1stars, topic_dist_list, 1)

cols = []
for i in range(1, numTopics+1):
    cols.append("Topic"+ str(i))
cols.append("Star")

topic_dist_test_1_2_3_4_5_df = pd.DataFrame(topic_dist_list, columns=cols)

def getSentiment(x):
    if x < 3.5:
        return 0
    else:
        return 1

topic_dist_train_1_2_3_4_5_df['Sentiment'] = topic_dist_train_1_2_3_4_5_df['Star'].map(getSentiment)
topic_dist_test_1_2_3_4_5_df['Sentiment'] = topic_dist_test_1_2_3_4_5_df['Star'].map(getSentiment)

############################Naive-Bayes########################################


vectorizer = TfidfVectorizer()

tfidfXtrain = vectorizer.fit_transform(all_text_train)
tfidfXtest = vectorizer.transform(all_text_test)

tfidfYtrain = topic_dist_train_1_2_3_4_5_df['Star']
tfidfYtest = topic_dist_test_1_2_3_4_5_df['Star']

clfs = [KNeighborsClassifier(), MultinomialNB(), LogisticRegression()]
clf_names = ['Nearest Neighbors', 'Multinomial Naive Bayes', 'Logistic Regression']

NBResults = {}
for (i, clf_) in enumerate(clfs):
    clf = clf_.fit(tfidfXtrain, tfidfYtrain)
    preds = clf.predict(tfidfXtest)
    
    precision = metrics.precision_score(tfidfYtest, preds,average='macro')
    recall = metrics.recall_score(tfidfYtest, preds,average='macro')
    f1 = metrics.f1_score(tfidfYtest, preds,average='macro')
    accuracy = accuracy_score(tfidfYtest, preds)
    report = classification_report(tfidfYtest,preds)
    matrix = metrics.confusion_matrix(tfidfYtest, preds)#starsGroup.groups.keys())
    mse = metrics.mean_squared_error(tfidfYtest, preds)
    data = {'precision':precision,
            'recall':recall,
            'f1_score':f1,
            'accuracy':accuracy,
            'clf_report':report,
            'clf_matrix':matrix,
            'y_predicted':preds,
            'mse':mse}
    NBResults[clf_names[i]] = data

cols = ['precision', 'recall', 'f1_score', 'accuracy','mse']
pd.DataFrame(NBResults).T[cols].T

for model, val in NBResults.items():
    print ('-------'+'-'*len(model))
    print ('MODEL:', model)
    print ('-------'+'-'*len(model))
    print ('The precision for this classifier is ' + str(val['precision']))
    print ('The recall for this classifier is    ' + str(val['recall']))
    print ('The f1 for this classifier is        ' + str(val['f1_score']))
    print ('The accuracy for this classifier is  ' + str(val['accuracy']))
    print ('The MSE is  ' + str(val['mse']))
    print ('Here is the classification report:')
    print (val['clf_report'])
    
    
########################LDA model####################
features = list(topic_dist_train_1_2_3_4_5_df.columns[:numTopics])
print(features)
x_train = topic_dist_train_1_2_3_4_5_df[features]
y_train = topic_dist_train_1_2_3_4_5_df['Star']

x_test = topic_dist_test_1_2_3_4_5_df[features]
y_test = topic_dist_test_1_2_3_4_5_df['Star'] 

clfs = [KNeighborsClassifier(), MultinomialNB(), LogisticRegression(), LDA(), QDA(), RandomForestClassifier(n_estimators=100, n_jobs=2), AdaBoostClassifier(n_estimators=100)]
clf_names = ['Nearest Neighbors', 'Multinomial Naive Bayes', 'Logistic Regression', 'LDA', 'QDA', 'Random Forest', 'AdaBoost']

LDAResults = {}
for (i, clf_) in enumerate(clfs):
    clf = clf_.fit(x_train, y_train)
    preds = clf.predict(x_test)
    
    precision = metrics.precision_score(tfidfYtest, preds,average='macro')
    recall = metrics.recall_score(tfidfYtest, preds,average='macro')
    f1 = metrics.f1_score(tfidfYtest, preds,average='macro')
    accuracy = accuracy_score(tfidfYtest, preds)
    report = classification_report(tfidfYtest,preds)
    matrix = metrics.confusion_matrix(tfidfYtest, preds)#starsGroup.groups.keys())
    mse = metrics.mean_squared_error(tfidfYtest, preds)
    
    data = {'precision':precision,
            'recall':recall,
            'f1_score':f1,
            'accuracy':accuracy,
            'clf_report':report,
            'clf_matrix':matrix,
            'y_predicted':preds,
            'mse':mse}
    
    LDAResults[clf_names[i]] = data
    
cols = ['precision', 'recall', 'f1_score', 'accuracy','mse']
pd.DataFrame(LDAResults).T[cols].T

for model, val in LDAResults.items():
    print ('-------'+'-'*len(model))
    print ('MODEL:', model)
    print ('-------'+'-'*len(model))
    print ('The precision for this classifier is ' + str(val['precision']))
    print ('The recall for this classifier is    ' + str(val['recall']))
    print ('The f1 for this classifier is        ' + str(val['f1_score']))
    print ('The accuracy for this classifier is  ' + str(val['accuracy']))
    print ('The MSE is  ' + str(val['mse']))
    print ('Here is the classification report:')
    print (val['clf_report'])



