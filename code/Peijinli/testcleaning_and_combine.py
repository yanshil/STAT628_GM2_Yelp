# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:57:49 2018

@author: Peiji
"""

file2 = 'testval_data.csv'
data2 = pd.read_csv(file2,sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],engine='python')
data2.text= process_reviews(data2.text)
data2.textlen = pd.Series([len(i) for i in data2.text])

#################### conbine test and train to do tf counting and category  ##############
#################### replace all the restaurantsDF to train dataset#######################
frames = [restaurantsDF.text, data2.text]
text = pd.concat(frames)
text.index = range(text.shape[0])

frames2 = [restaurantsDF.categories,data2.categories]
cate = pd.concat(frames2)
cate.index = range(cate.shape[0])

text= process_reviews(text)

nofeature = 1000000
tf_vectorizer = CountVectorizer(max_features= nofeature)
finaltextTF = tf_vectorizer.fit_transform(text)

#################### split the train and test to do the model fitting####################

finaltrain_textTF = finaltextTF[range(0,1500000),]
finaltest_textTF = finaltextTF[range(1500000,2516664),]#1016664+1500000   vacat waitress seem annoy want food know want

finalcategory = getsp_category(cate)[0]
finaltrain_cat = category[range(0,1500000),]
finaltest_cat = category[range(1500000,2516664),]

#################### combine other feature ############################################
finalX_train2 = hstack((finaltrain_textTF,np.array(restaurantsDF['longitude'])[:,None],
                  np.array(restaurantsDF.['latitude'])[:,None],finaltrain_cat,np.array(restaurantsDF.textlen)[:,None]))

finalX_test2 = hstack((finaltest_textTF,np.array(data2['longitude'])[:,None],
                  np.array(data2['latitude'])[:,None],finaltest_cat,np.array(data2.textlen)[:,None]))

################### fitting svm model #################################################
weight = getproportion(finalX_train2,'stars')
wclf = svm.SVC(kernel='linear', class_weight=weight)
wclf.fit(X_train2, restaurantsDF.stars)
finalpredY = wclf.predict(finalX_test2)
##################print output#########################################################
pd.DataFrame(finalpredY).to_csv('predict.csv',index = False)