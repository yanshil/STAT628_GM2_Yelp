#!/usr/bin/env python3

"""
run_skML.py: run Machine Learning Models of sklearn with tidied input
"""

__author__ = "Yanshi Luo", "Peijin Li"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"


import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
import extra_features2csr
import process_text
import sklearn_pack

"""
Scripts for features generating and Models
"""

train_filename = 'train_data.csv'
test_filename = 'testval_data.csv'

isUnitTest = True

trainDF = pd.read_csv(train_filename)
testDF = pd.read_csv(test_filename)

if isUnitTest:
    trainDF = trainDF.head(500)
    testDF = testDF.head(10)

n_train = trainDF.shape[0]
n_test = testDF.shape[0]

"""
Get TF-IDF from Text
"""

num_feature = 1000000

train_tfVec = CountVectorizer(max_features=num_feature)
final_train_textTF = train_tfVec.fit_transform(process_text.process_reviews(trainDF.text))
# Get Vocalbulary for generating sparse matrix
train_features = train_tfVec.get_feature_names()

test_tfVec = CountVectorizer(vocabulary=train_features)
final_test_textTF = test_tfVec.fit_transform(process_text.process_reviews(testDF.text))

print(final_train_textTF.shape)
print(final_test_textTF.shape)

"""
Get Sparse Matrix from Categories 
"""
comment_categories = [trainDF.categories, testDF.categories]
categories = pd.concat(comment_categories)

final_category = process_text.get_category_sp(categories)
# a = csr_matrix(final_category.values)
final_train_category = csr_matrix(final_category.iloc[0:n_train, ].values)
final_test_category = csr_matrix(final_category.iloc[n_train: n_train + n_test, ].values)


"""
Extra Features
"""

final_train_extra_features = extra_features2csr.get_extra_features(trainDF)
final_test_extra_features = extra_features2csr.get_extra_features(testDF)


"""
Combine all features to get final Train-Test set
"""
finalX_train2 = hstack([final_train_textTF, final_train_category, final_train_extra_features])
finalX_test2 = hstack([final_test_textTF, final_test_category, final_test_extra_features])


print(finalX_train2.shape)
print(finalX_test2.shape)


"""
Fit RF model
"""
# final_RF_pred = sklearn_pack.random_forest(finalX_train2, trainDF.stars, finalX_test2, n_parallel=4)
# final_RF_pred.to_csv('predict_RF.csv', index=False)

"""
Fit Decision Tree model
"""
# final_DT_pred = sklearn_pack.decision_tree(finalX_train2, trainDF.stars, finalX_test2)
# final_DT_pred.to_csv('predict_DTree.csv', index=False)

"""
Fit NN model
"""
final_NN_pred = sklearn_pack.neural_network(finalX_train2, trainDF.stars, finalX_test2)
final_NN_pred.to_csv('predict_NN.csv', index=False)