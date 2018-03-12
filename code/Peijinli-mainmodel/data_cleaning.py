# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:41:28 2018

@author: Peiji
"""
import nltk
import numpy as np
import pandas as pd
import pylab
import re
import scipy as sp
import seaborn
import gensim
import sklearn

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

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


###########function to extract word stem ####################################################

ps = PorterStemmer()

########################################set up stopwords####################################

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

########################################extract abbreviation##################################
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

################################################## throw too large or too short comment####
def subset_givenlength(data,minReviewLen,maxReviewLen,subsize):
    t= data.sample(n=subsize, frac=None, replace=False, weights=None, random_state=None, axis=0)
    print("Number of rows selected:",len(t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]))
    subset = t[t.text.str.len() > minReviewLen][t.text.str.len() < maxReviewLen]
    return subset


################################################### clean whole comments list###############
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

################################################### category clean######################

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
    
##########################################  read data  ##################################################
#path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'
#os.chdir(path)
from sklearn.cross_validation import train_test_split  

###read data,split in train test
file = 'train_data.csv'
data = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
restaurantsDF = subset_givenlength(data,0,10000,15000)
restaurantsDF.index = range(restaurantsDF.shape[0])
restaurantsDF.text= process_reviews(restaurantsDF.text)

###########################################   TF-IDF   ######################################
tfidf_vectorizer = TfidfVectorizer()
textTFIDF = tfidf_vectorizer.fit_transform(restaurantsDF.text)
textTFIDF_feature_names = tfidf_vectorizer.get_feature_names()

tf_vectorizer = CountVectorizer()
textTF = tf_vectorizer.fit_transform(restaurantsDF.text)
textTF_feature_names = tf_vectorizer.get_feature_names()




###########################################  Cross-validation ###############################
train, test = train_test_split(restaurantsDF, test_size=0.2)
train_textTFIDF = textTFIDF[train.index,]
test_textTFIDF = textTFIDF[test.index,]
########################################## categories sparse matrix#######################

category = getsp_category(restaurantsDF.categories)[0]
train_cat = category[train.index,]
test_cat = category[test.index,]
train_len = restaurantsDF.textlen[train.index]
test_len = restaurantsDF.textlen[test.index]
#############################################################################################


#Ithink you should build the TF-IDF for the whole dataset
#IF using CV to determine the MSE, sparse matrix can be access by sparsematrix[rowlist,:]
#using  train_test_split() split training adn test set
# also doing subset would keep the index, so you need resign index for subset
# I already include all the step in the above code

