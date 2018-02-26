# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:54:20 2018

@author: Peiji
"""
import sys
import pandas as pd
import numpy as np
import os
import csv
import sklearn
import numpy.linalg as linalg
from sklearn.feature_extraction.text import CountVectorizer  
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.random_projection import sparse_random_matrix
from matplotlib import pyplot as plt
path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'
os.chdir(path)
os.listdir(path)

comments = pd.Series.from_csv('train_comment_deletestop.csv',sep=",")
corpus = comments
num = 0
for i in comments:
    if type(i)=='float':
        print(num)
    num += 1
def get_sparse_matrix(corpus):
    #input comment as list
    #return words counts ,tfidf sparse matrix
vectorizer = CountVectorizer()  
X = vectorizer.fit_transform(corpus.values.astype('U'))#words count sparse matrix
word = vectorizer.get_feature_names() #words result
word = pd.DataFrame(word)
word.to_csv('words.csv')
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(X) #tfidf is tf-idf sparse matrix
#    return [X,tfidf]

def PCA(tfidf):
A = tfidf
N=  A.shape[1]
C=((A.T*A -(sum(A).T*sum(A)/N))/(N-1)).todense()
V=np.sqrt(np.mat(np.diag(C)).T*np.mat(np.diag(C)))
COV = np.divide(C,V+1e-119)
eig_vals, eig_vecs = linalg.eig(COV)
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# return [eig_vals,eig_vecs,var_exp,cum_var_exp]

biggest = eig_vecs[0]
snd = eige_vecs[1]
#####
#所以可以用前2个eigenvector里系数大于1词的作为变量

