# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 17:27:08 2018

@author: Wen
"""


import pandas as pd
import nltk
from nltk.corpus import stopwords
import time
import os
import re

filename = 'train_data.csv'

t1 = time.time()
reviews = pd.read_csv('train_data.csv')
time.time() - t1   #看看用了多少秒

cityname = reviews['city'].unique()
reviews.shape

reviews.head()

# Write the data frame to csv

rev_200 = reviews.iloc[0:200,]
rev_200.to_csv("first200.csv", index = False)
rev_200

# Check one example
rev_200.loc[0, ['stars', 'text']]

sentence = rev_200.loc[0, 'text']
sentence

import re
# Regular expression
sen_let = re.sub('[^a-zA-Z]',' ', sentence)
sen_let

# Noise Removal

stops = set(stopwords.words("english"))
sen_new = [w for w in sen_let.lower().split() if not w in stops]  
print(" ".join(sen_new))
print("\norginal sentence:\n")
print(sen_let)
