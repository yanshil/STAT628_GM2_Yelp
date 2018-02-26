# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:49:27 2018

@author: Peiji
"""
import sys
import pandas as pd
import numpy as np
import os
import csv
import nltk
import csv
import json
import gensim
from scipy import sparse
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
from gensim.models import Word2Vec, KeyedVectors

stem = PorterStemmer()
lem = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

#separate comment and category
def modtrain_data():
    file = 'train_data.csv'
    df = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
    df.columns
    ser1 = df['text']
    ser2 = df['categories']
    #df_comment = pd.concat([ser1])
    #df_cate = pd.concat([ser2])
    ser1.to_csv('comment.csv')
    ser2.to_csv('category.csv')
    print('traindata has been modified')


#for each comment, filter stop words, transfer to lowercase
def filterStopword(comment):
    words = word_tokenize(comment)
    wordsFiltered = []
    for w in words:
        if w not in stopWords:
            wordsFiltered.append(w.lower())
    Stopnum = len(words) - len(wordsFiltered)
    freqDict = dict(nltk.FreqDist(wordsFiltered))
    return [Stopnum,freqDict]
    


#for comments list return [[stop#,WordsMatrix]... ...]
#WordsMatrix is dictionary With key = words and value = frequency
def wordDataset(comments):
    Stopnum_Dataset = []
    WordsMatrix = []
    num = 0
    for i in comments:
        num += 1
        print(num)
        Stopnum_Dataset.append(filterStopword(i)[0]) 
        WordsMatrix.append(filterStopword(i)[1])
    return [Stopnum_Dataset,WordsMatrix] 
        



#modtrain_data()
path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2\\data_cleaning'
os.chdir(path)
os.listdir(path)
#comments = pd.Series.from_csv('comment.csv',sep=",")
#Dataset = wordDataset(comments)
#Stopnum = pd.Series(Dataset[0])
#Stopnum.to_csv('StopNum.csv')
#A=Dataset[1]
#with open("file_temp.json", "w") as f:
#    json.dump(A, f)
#read .json file

with open("file_temp.json", "r") as f:
    comments_dict = json.load(f)

#t:sentences = comments_dict[1:5]
N = len(comments_dict)


wordsIDF = {}### for each words {key = word : value = occur times}
wordsdict = []### immutable words
for i in comments_dict:
    for j in i.keys():
        wordsdict.append(j)
        
wordsbag_key = set(wordsdict)

num = 0
for i in wordsbag_key:
    num += 1
    print(num)
    value = 0
    for j in comments_dict:
        if i in j.keys():
            value += 1
    wordsIDF[i] = value/N

with open("words_IDF.json", "w") as f:
    json.dump(wordsIDF, f)
    
    
#wordTF = {}
#num = 0
#for i in comments_dict:
#    print(num)
#    total = sum(i.values())
#    frq_dic = i
#    for j in i.keys():
#        frq_dic[j] = i[j]/total
#    wordTF[num] = frq_dic
#    num += 1
    
#import json
#with open('outputfile.json', 'w') as fout:
#    json.dump(wordTF, fout)
    
with open("outputfile.json", "r") as f:
    out = json.load(f)
