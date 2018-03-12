# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 20:43:29 2018

@author: Peiji
"""
from collections import Counter

yelp_categories = train['categories']
yelp_categories_tidy = [re.sub("\'", '', x.strip("[]")).split(',') for x in data.categories]
test_yelp_categories_tidy  = [re.sub("\'", '', x.strip("[]")).split(',') for x in data2.categories]
#categories_counter = Counter()
#for x in yelp_categories_tidy:
#    categories_counter.update(x)
#categories_dict = dict(categories_counter)
#cate = pd.DataFrame(list(categories_dict.items()),columns=['Name', 'Counts'])
#cate.sort_values(by= ['Counts'], ascending=False)
#catelist = list(cate[cate['Counts']>40000]['Name'])

file = open("cattrain.txt","w")
for i in range(train.shape[0]):
    j = ('').join(('__label__',str(train.stars[i]),' ',(' ').join(yelp_categories_tidy[i])))
    file.write(j)
    file.write('\n')
file.close()

file = open("cattrain_pred.txt","w")
for i in range(train.shape[0]):
    j = (' ').join(yelp_categories_tidy[i])
    file.write(j)
    file.write('\n')
file.close()

file = open("cattest.txt","w")
for i in range(train.shape[0],train.shape[0]+test.shape[0]):
    j = (' ').join(yelp_categories_tidy[i])
    file.write(j)
    file.write('\n')
file.close()


file = open("catfinaltest.txt","w")
for i in range(data2.shape[0]):
    j = (' ').join(test_yelp_categories_tidy [i])
    file.write(j)
    file.write('\n')
file.close()

file = open("catfinaltrain.txt","w")
for i in range(data.shape[0]):
    j = ('').join(('__label__',str(data.stars[i]),' ',(' ').join(yelp_categories_tidy[i])))
    file.write(j)
    file.write('\n')
file.close()


file = open("catfinaltrain_pred.txt","w")
for i in range(data.shape[0]):
    j = (' ').join(yelp_categories_tidy[i])
    file.write(j)
    file.write('\n')
file.close()


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


#cvtrain,cvtestcat,finaltest
cvtrain,cvtestcat,finaltest


