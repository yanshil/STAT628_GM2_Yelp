
"""
doc2vec_model.py: Train Model with doc2vec
"""

__author__ = "Yanshi Luo", "Peijin Li"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"

from gensim.models import Doc2Vec
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack

import process_text
import extra_features2csr
from sklearn_pack import random_forest

# import os
# path = 'C:\\Users\\kdrob\\Downloads'
# os.chdir(path)

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

model = Doc2Vec.load('./yelp.doc2vec')
# model = Doc2Vec.load('C:\\Users\\kdrob\\OneDrive - UW-Madison\\'
#                      'WiscTextbook\\Spring 2018\\STAT628\\Module2\\'
#                      'STAT628_GM2_Yelp\\code\\Yanshi\\yelp.doc2vec')


def get_text_vec(data):
    data['tokens'] = [review.split() for review in process_text.process_reviews(data.text)]

    M = np.empty((0, model.vector_size))

    for i in range(0, n_train):
        vec = model.infer_vector(data.tokens[i])
        M = np.concatenate((M, vec[:, None].T), axis=0)

    text_vec = csr_matrix(M)

    return text_vec


train_text_vec = get_text_vec(trainDF)
test_text_vec = get_text_vec(testDF)

print(train_text_vec.shape)
print(test_text_vec.shape)
# model.infer_vector(trainDF.tokens[0])
#
# model.docvecs[trainDF.stars[0]]
# model.wv.vocab

final_train_extra_features = extra_features2csr.get_extra_features(trainDF)
final_test_extra_features = extra_features2csr.get_extra_features(testDF)

finalX_train2 = hstack([train_text_vec, final_train_extra_features])
finalX_test2 = hstack([test_text_vec, final_test_extra_features])

print(finalX_train2.shape)
print(finalX_test2.shape)


final_RF_pred = random_forest(finalX_train2, trainDF.stars, finalX_test2, n_parallel=4)
final_RF_pred.to_csv('predict_RF_doc2vec.csv', index=False)
