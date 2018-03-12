#!/usr/bin/env python3

"""
run_skML.py: run Machine Learning Models of sklearn with tidied input
"""

__author__ = "Yanshi Luo", "Peijin Li"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"


import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import extra_features2csr
import process_text
import sklearn_pack
import load_vocab
import sys

"""
Scripts for features generating and Models
"""

train_filename = 'train_data.csv'
test_filename = 'testval_data.csv'

isUnitTest = True
isCV = True
isRebuild = False

trainDF = pd.read_csv(train_filename)

if isCV:
    testDF = trainDF[450000:600000]
    trainDF = trainDF[0:450000]
else:
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
good_cate_list = ['Delis', 'Polish', 'Vegan', 'French', 'Peruvian']
bad_cate_list = ['Chicken_Wings', 'Fast_Food', 'Buffets', 'Tex-Mex', 'Burgers']
full_list = [x.lower() for x in good_cate_list + bad_cate_list]

final_train_category = load_vocab.get_category_features_sp(trainDF.categories, full_list)
final_test_category = load_vocab.get_category_features_sp(testDF.categories, full_list)

# comment_categories = [trainDF.categories, testDF.categories]
# categories = pd.concat(comment_categories)
#
# final_category = process_text.get_category_sp(categories)
# # a = csr_matrix(final_category.values)
# final_train_category = csr_matrix(final_category.iloc[0:n_train, ].values)
# final_test_category = csr_matrix(final_category.iloc[n_train: n_train + n_test, ].values)

"""
Previous Model Input
"""
if isRebuild:
    train_label_filename = ''
    train_label = extra_features2csr.label_input(train_label_filename)
    test_label_filename = ''
    test_label = extra_features2csr.label_input(test_label_filename)


"""
Extra Features
"""

final_train_extra_features = extra_features2csr.get_extra_features(trainDF)
final_test_extra_features = extra_features2csr.get_extra_features(testDF)

"""
Combine all features to get final Train-Test set
"""
if not isRebuild:
    finalX_train2 = hstack([final_train_textTF, final_train_category, final_train_extra_features])
    finalX_test2 = hstack([final_test_textTF, final_test_category, final_test_extra_features])

if isRebuild:
    finalX_train2 = hstack([final_train_textTF, final_train_category, final_train_extra_features, train_label])
    finalX_test2 = hstack([final_test_textTF, final_test_category, final_test_extra_features, test_label])


print(finalX_train2.shape)
print(finalX_test2.shape)

"""
Fit RF model
"""
# final_RF_pred = sklearn_pack.random_forest(finalX_train2, trainDF.stars, finalX_train2, n_parallel=4)
# final_RF_pred.to_csv(sys.argv[0] + 'predict_RF_cv_train.csv', index=False)
#
# if isCV:
#     final_RF_pred2 = sklearn_pack.random_forest(finalX_train2, trainDF.stars, finalX_test2, n_parallel=4)
#     final_RF_pred2.to_csv(sys.argv[0] + 'predict_RF_cv_test.csv', index=False)

sklearn_pack.random_forest(finalX_train2, trainDF.stars, finalX_train2, n_parallel=4,
                           write_csv=True, write_filename=sys.argv[0] + 'predict_RF_cv_train.csv')

sklearn_pack.random_forest(finalX_train2, trainDF.stars, finalX_test2, n_parallel=4,
                           write_csv=True, write_filename=sys.argv[0] + 'predict_RF_cv_test.csv')

"""
Fit Decision Tree model
"""
# final_DT_pred = sklearn_pack.decision_tree(finalX_train2, trainDF.stars, finalX_train2)
# final_DT_pred.to_csv(sys.argv[0] + 'predict_DTree_cv_train.csv', index=False)
#
# if isCV:
#     final_DT_pred2 = sklearn_pack.random_forest(finalX_train2, trainDF.stars, finalX_test2)
#     final_DT_pred2.to_csv(sys.argv[0] + 'predict_DTree_cv_test.csv', index=False)

"""
Fit NN model
"""
# final_NN_pred = sklearn_pack.neural_network(finalX_train2, trainDF.stars, finalX_test2)
# final_NN_pred.to_csv('predict_NN.csv', index=False)
