# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 19:45:18 2018

@author: Peiji
"""
#https://www.jianshu.com/p/d50a14541d01

path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2\\data_cleaning'
os.chdir(path)
os.listdir(path)
import re
from textblob import TextBlob
comments = pd.Series.from_csv('comment.csv',sep=",")
segement = []
num = 0
for i in comments:
    num += 1
    print(num)
    seg = []
    abs_seg = []
    blob = TextBlob(i)
    for j in range(len(blob.sentences)):
        abs_seg.append(abs(blob.sentences[j].sentiment[0]))
        seg.append(blob.sentences[j].sentiment[0])
        res = seg[abs_seg.index(max(abs_seg))]
    segement.append(res)
    
aaa = segement
aaa = pd.Series(segement)
aaa.to_csv('blob_segement_max_version.csv')

file = 'train_data.csv'
df = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
df.columns
df.stars

a = [aaa,df.stars]
a = pd.DataFrame(a)
b = a.transpose()
b.to_csv('blob_segement_plus_stars_max_version.csv')
