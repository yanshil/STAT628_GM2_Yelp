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
path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'
os.chdir(path)
os.listdir(path)
#print(sys.version)
file = 'train_data.csv'
df = pd.read_csv(file,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
df.columns
ser1 = df['text']
ser2 = df['categories']
df_comment = pd.concat([ser1])
df_cate = pd.concat([ser2])
df_comment.to_csv('comment.csv')
df_cate.to_csv('category.csv')

