# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 17:04:08 2018

@author: Peiji
"""
def RB(data):
    tex = []
    num = 0
    for sentense in data:
        num += 1
        print(num)
        res = []
        a = word_tokenize(sentense)
        b = nltk.pos_tag(a)
        res = []
        for i in b:
            if i[1] == 'RB':
                res.append(i[0])
        tex.append(res)
    return tex
    
train1 = RB(dattext)
train2 = [(" ").join(i) for i in train1]
pd.DataFrame(train2).to_csv('cvtrainRB.csv')

dattext2 = process_reviews(test.text)
test1 = RB(dattext2)
test1[2]
test2 = [(" ").join(i) for i in test1]
pd.DataFrame(test2).to_csv('cvtestRB.csv')


finaltext2 = process_reviews(data2.text)
final2 = RB(finaltext2)
final3 = [(" ").join(i) for i in final2]
pd.DataFrame(final3).to_csv('finalRB.csv')


nofeature = 200
tf_vectorizer = CountVectorizer(max_features= nofeature)
textTF = tf_vectorizer.fit_transform(train2)
textTF_feature_names = tf_vectorizer.get_feature_names()

tf = CountVectorizer(max_features= nofeature,vocabulary = textTF_feature_names)
test_textTF = tf.fit_transform(test2)



def getproportion(dataset, labelname):
    weight = {}
    total = sum(dataset.groupby(labelname).size())
    for i in dataset.groupby(labelname).size().index:
        weight[i] = train.groupby(labelname).size().get(i)/total
    return weight


from scipy.sparse import hstack
from sklearn import svm
X_train = textTF
X_test = test_textTF

weight = getproportion(train,'stars')
wclf = svm.SVC(kernel='linear', class_weight=weight)
wclf.fit(X_train, train.stars)
predY = wclf.predict(X_test)
mean_squared_error(test.stars, predY)#1.679



