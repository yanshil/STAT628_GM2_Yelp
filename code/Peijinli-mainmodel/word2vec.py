# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:21:06 2018

@author: Peiji
"""

def process_reviews(dirty_data_set):
    clean_data_set = []
    for review in dirty_data_set:
        review = decontracted(review)
        # Remove punctuations        
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
        texts = [word for word in review.lower().split() if word not in stoplist]
        try:
            clean_data_set.append(texts)
        except:
            pass
    return clean_data_set


#model.save(fname)
#model = Word2Vec.load(fname)
import os
import gensim
from gensim import models
from gensim.models import Word2Vec, KeyedVectors
from gensim.models import KeyedVectors

from sklearn.cross_validation import train_test_split
subdata = subset_givenlength(data,0,10000,1000)
train, test = train_test_split(subdata, test_size=0.2)

train.text = process_reviews(train.text)
test.text = process_reviews(test.text)
train.index = range(train.shape[0])
test.index = range(test.shape[0])


sentences = train.text
model = Word2Vec(sentences, size=1000)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))


#test.text = process_reviews(test.text)
#model.score(test.text[1])
#model.wv.doesnt_match(test.text[0])

train.text
trainvec = []
useless = []
for i in range(len(train.text)):
    ce = []
    for j in train.text[i]:
        number = 0
        if j in w2v.keys():
            ce.append(model.wv[j])
            number += 1
    a=np.mean(ce,axis=0)
    a.tolist()
    try:
        len(a)== 1000 
        trainvec.append(a)
    except:
        useless.append(i)


test.text
testvec = []
ul = []
for i in range(len(test.text)):
    ce = []
    for j in test.text[i]:
        number = 0
        if j in w2v.keys():
            ce.append(model.wv[j])
            number += 1
    a=np.mean(ce,axis=0)
    a.tolist()
    try:
        len(a)== 1000 
        testvec.append(a)
    except:
        ul.append(i)


def getproportion(dataset, labelname):
    weight = {}
    total = sum(dataset.groupby(labelname).size())
    for i in dataset.groupby(labelname).size().index:
        weight[i] = train.groupby(labelname).size().get(i)/total
    return weight


from sklearn import svm
X_train = trainvec

X_test = testvec
weight = getproportion(train,'stars')

wclf = svm.SVC(kernel='linear', class_weight=weight)
wclf.fit(X_train, train.stars.drop(useless))
predY = wclf.predict(X_test)
hatY = wclf.predict(X_train)
mean_squared_error(test.stars.drop(ul), predY)
# 
# test1w: size :100 :3.3515 
# 3w size :1000 1.302
#10w size :1000 0.9235961798089904

################################tree for final#############################


def random_forest(finalX_train, finalY_train, finalX_test, n_parallel=1):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, n_jobs=n_parallel)
    clf = clf.fit(finalX_train, finalY_train)
    finalY_pred = clf.predict(finalX_test)
    return pd.DataFrame(finalY_pred)


categories = pd.concat(train.categories,test.categories)
categories.index = range(categories.shape[0])
final_category = get_category_sp(categories)

n_train = trainDF.shape[0]
n_test = testDF.shape[0]

final_train_category = final_category.iloc[0:n_train, ]
final_test_category = final_category.iloc[n_train: n_train + n_test, ]
final_test_category.index = range(final_test_category.shape[0])

final_train_extra_features = train[['city','name','latitude','longitude']]
final_test_extra_features = test[['city','name','latitude','longitude']]
    
finalX_train = pd.concat([X_train, final_train_category, final_train_extra_features], axis=1)
finalX_test= pd.concat([X_test, final_test_category, final_test_extra_features], axis=1)



final_RF_pred = random_forest(finalX_train, train.stars.drop(useless), finalX_test, n_parallel=4)
final_RF_pred.to_csv('predict_RF.csv', index=False)



 
hatY
type(predY)

