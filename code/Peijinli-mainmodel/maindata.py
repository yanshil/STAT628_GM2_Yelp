# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 18:04:16 2018

@author: Peiji
"""

import os
import pandas as pd
path = 'C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2'

################################## read data##################################
os.chdir(path)
data = pd.read_csv('train_data.csv',sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
data2 = pd.read_csv('testval_data.csv',sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')

#data[0:100].text.to_csv('100train.csv',index=False)
from sklearn.cross_validation import train_test_split
#subdata = subset_givenlength(data,0,10000,1000000)
train, test = data[0:450000],data[450000:600000]

#train_test_split(subdata, test_size=0.2)
train.index = range(train.shape[0])
test.index = range(test.shape[0])

################################## read fastresult ##################################

train_pred = []
file = open("cvtrain_hat.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    train_pred.append(int(i.encode('utf-8')))
file.close()
print(mean_squared_error(train.stars,train_pred)) #0.5274911111111111
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


train_cat = []
file = open("cvtraincat_hat.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    train_cat.append(int(i.encode('utf-8')))
file.close()
print(mean_squared_error(train.stars,train_cat)) #3.110682222222222
train_cat_pred = pd.DataFrame(train_cat,columns = ['catscore'])


test_cat = []
file = open("cvtestcat_pred.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    test_cat.append(int(i.encode('utf-8')))
file.close()
print(mean_squared_error(test.stars,test_cat)) #3.1150866666666666
test_cat_pred = pd.DataFrame(test_cat,columns = ['catscore'])


final_cat = []
file = open("finaltestcat_pred.txt")
for i in file.readlines():
    #i = i.encode('utf-8')
    final_cat.append(int(i.encode('utf-8')))
file.close()
final_cat_test = pd.DataFrame(final_cat,columns = ['catscore'])



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


pd.concat([train_cat_pred,train_pred],axis = 1).to_csv("m2_train.csv")
pd.concat([test_cat_pred,test_pred],axis = 1).to_csv("m2_test.csv")
pd.concat([final_cat_test,final_test],axis = 1).to_csv('finaltest.csv')



#############################################test#############################################

train2 = pd.read_csv("m2_train.csv")
test2 = pd.read_csv("m2_test.csv")
final2 = pd.read_csv('finaltest.csv')
del train2['Unnamed: 0']
del test2['Unnamed: 0']
del final2['Unnamed: 0']
print(train2.shape)
print(test2.shape)
print(final2.shape)

train2['city'] = train.city
test2['city'] = test.city
final2['city'] = data2.city



