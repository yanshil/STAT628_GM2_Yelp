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

########################################set up stopwords#####################################

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


################################################### clean whole comments list###########################
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

################################################### category clean#######################################

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

######################################　read data,split in train test　###################################
file = 'train_data.csv'
data = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
data2 = pd.read_csv('testval_data.csv',sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')

train, test = data[0:450000],data[450000:600000]
train.index = range(train.shape[0])
test.index = range(test.shape[0])


################################## read fastTest  text score result ###################################################

train_pred = []
file = open("cvtrain_hat.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    train_pred.append(int(i.encode('utf-8')))
file.close()
print(mean_squared_error(train.stars,train_pred)) 
train_pred = pd.DataFrame(train_pred)

test_pred = []
file = open("cvtest_pred.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    test_pred.append(int(i.encode('utf-8')))
file.close()
test_pred = pd.DataFrame(test_pred)

final_test = []
file = open("test_pred.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    final_test.append(int(i.encode('utf-8')))
file.close()
final_test = pd.DataFrame(final_test)

################################## read fastTest categories score result ###################################################

train_cat = []
file = open("cvtraincat_hat.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    train_cat.append(int(i.encode('utf-8')))
file.close()
print(mean_squared_error(train.stars,train_cat)) 
train_cat_pred = pd.DataFrame(train_cat,columns = ['catscore'])


test_cat = []
file = open("cvtestcat_pred.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    test_cat.append(int(i.encode('utf-8')))
file.close()
print(mean_squared_error(test.stars,test_cat))
test_cat_pred = pd.DataFrame(test_cat,columns = ['catscore'])


final_cat = []
file = open("finaltestcat_pred.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    final_cat.append(int(i.encode('utf-8')))
file.close()
final_cat_test = pd.DataFrame(final_cat,columns = ['catscore'])
################################## good or bad categories ###################################################################

goodcatelist = ['Delis','Polish','Vegan','French','Peruvian'] 
badcatelist = ['Chicken Wings','Fast Food','Buffets','Tex-Mex','Burgers']


train_catelist = []
for i in range(train.shape[0]):
    cate_1 = [0]*2
    for j in yelp_categories_tidy[i]:
        if j in goodcatelist:
            cate_1[0] += 1
        if j in badcatelist:
            cate_1[1] += 1
    train_catelist.append(cate_1)

test_catelist = []
for i in range(train.shape[0],train.shape[0]+test.shape[0]):
    cate_1 = [0]*2
    for j in yelp_categories_tidy[i]:
        if j in goodcatelist:
            cate_1[0] += 1
        if j in badcatelist:
            cate_1[1] += 1
    test_catelist.append(cate_1)

finaltest = []
for i in range(data2.shape[0]):
    cate_1 = [0]*2
    for j in test_yelp_categories_tidy[i]:
        if j in goodcatelist:
            cate_1[0] += 1
        if j in badcatelist:
            cate_1[1] += 1
    finaltest.append(cate_1)

trcat = pd.DataFrame(train_catelist,columns = ['good','bad'])
ttcat = pd.DataFrame(test_catelist,columns = ['good','bad'])
ft = pd.DataFrame(finaltest,columns = ['good','bad'])


print(len(train_catelist ))
print(len(test_catelist ))
print(len(finaltest ))

################################## other feature ############################################################################

def count_upper_word(text):
    words = text.split()
    upper_list = [word for word in words if word.isupper()]
    # Remove single capital letter
    count = len([word for word in upper_list if len(word) > 1])

    return count

traintext = process_reviews(train.text)
testtext = process_reviews(test.text)
finaltesttext = process_reviews(data2.text)


train_pred["text_length"] = pd.Series([len(i) for i in train.text])
resttrain_len = pd.Series([len(i) for i in traintext])
train_pred["rest_length"] = train_pred["text_length"] - resttrain_len 
train_pred["num_upper_words"] = pd.Series([count_upper_word(x) for x in train.text])
train_pred["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in train.text])
train_pred["question_mark"] = pd.Series([len(re.findall(r'\?', x)) for x in train.text])
train_pred["dollar"] = pd.Series([len(re.findall(r'\$', x)) for x in train.text])
train_pred["precent"] = pd.Series([len(re.findall(r'\%', x)) for x in train.text])
train_pred["facebad"] = pd.Series([len(re.findall(r'\:\(', x)) for x in train.text])
train_pred["facegood"] = pd.Series([len(re.findall(r'\:\)', x)) for x in train.text])
train_pred['superb']  =  pd.Series([len(re.findall(r'superb', x)) for x in train.text])
train_pred['goodcat']  = trcat.good
train_pred['badcat'] = trcat.bad
train_pred['stars'] = train.stars
train_pred['city'] = train.city



test_pred["text_length"] = pd.Series([len(i) for i in test.text])
resttest_len = pd.Series([len(i) for i in testtext])
test_pred["rest_length"] = test_pred["text_length"] - resttest_len
test_pred["num_upper_words"] = pd.Series([count_upper_word(x) for x in test.text])
test_pred["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in test.text])
test_pred["question_mark"] = pd.Series([len(re.findall(r'\?', x)) for x in test.text])
test_pred["dollar"] = pd.Series([len(re.findall(r'\$', x)) for x in test.text])
test_pred["precent"] = pd.Series([len(re.findall(r'\%', x)) for x in test.text])
test_pred["facebad"] = pd.Series([len(re.findall(r'\:\(', x)) for x in test.text])
test_pred["facegood"] = pd.Series([len(re.findall(r'\:\)', x)) for x in test.text])
test_pred['superb']  =  pd.Series([len(re.findall(r'superb', x)) for x in test.text])
test_pred['goodcat']  = ttcat.good
test_pred['badcat'] = ttcat.bad
test_pred['stars'] = test.stars
test_pred['city'] = test.city




final_test["text_length"] = pd.Series([len(i) for i in data2.text])
restfinaltest_len = pd.Series([len(i) for i in finaltesttext])
final_test["rest_length"] = final_test["text_length"] - restfinaltest_len
final_test["num_upper_words"] = pd.Series([count_upper_word(x) for x in data2.text])
final_test["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in data2.text])
final_test["question_mark"] = pd.Series([len(re.findall(r'\?', x)) for x in data2.text])
final_test["dollar"] = pd.Series([len(re.findall(r'\$', x)) for x in data2.text])
final_test["precent"] = pd.Series([len(re.findall(r'\%', x)) for x in data2.text])
final_test["facebad"] = pd.Series([len(re.findall(r'\:\(', x)) for x in data2.text])
final_test["facegood"] = pd.Series([len(re.findall(r'\:\)', x)) for x in data2.text])
final_test['superb']  =  pd.Series([len(re.findall(r'superb', x)) for x in data2.text])
final_test['goodcat']  = ft.good
final_test['badcat'] = ft.bad
final_test['city'] = data2.city


pd.concat([train_cat_pred,train_pred],axis = 1).to_csv("m2_train.csv")
pd.concat([test_cat_pred,test_pred],axis = 1).to_csv("m2_test.csv")
pd.concat([final_cat_test,final_test],axis = 1).to_csv('finaltest.csv')