#!/usr/bin/env python3

"""sklearn_pack.py: Do Model training with large sparse matrix with sklearn"""

__author__ = "Yanshi Luo", "Peijin Li"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.stem import PorterStemmer
import string

"""
Text Processing Rules @ Peijin
"""

ps = PorterStemmer()

customized_stopwords_list = {'ll', 'are', 'theirs', 'up', 'do', 'have', 'who', 'few', 'needn', 'yourselves', 'has',
                             'under', 'ain', 'the', 'should', 'y', 'might', "must", 'was', 'had', 'she', 'is',
                             'through',
                             'himself', 'their', 'ours', 'm', 'and', 'am', 'against', 'his', 'from', 'mustn', 'off',
                             'her',
                             "will", 'myself', 'as', "did", "is", 'themselves', 'o', 'of', 'them', 'does', 'i', 'a',
                             'by',
                             't', 'had', 'it', 'after', "should've", 'was', 'did', 'my', 'into', 'they', 'such', 'but',
                             'if',
                             'hers', 'with', 'your', 'than', "had", 'did', "could", "she's",
                             "might", 'has', 'each', 'these', 'our', 'will', 'those', 'can', 'he', 'over', 'could',
                             'having', 'below', 'between', 'own', 'until', 'about', 'all', 'being', 'why', 'should',
                             'most', 're', 'we', 'doing', 'at', 'because', 's', 'does', 'now', 'other', 'down',
                             'ourselves', 'so', 'you', 'were', 'while', 'to', 'here', 'me', "you've", 'its', 'herself',
                             'further', 'too', 'isn', "you're", 'were', "was", 'some', 'in', 'been', "it's", 'or',
                             'are', 'nor', "have", 'same', 'before', 'won', 'when', 'more', 'this', 'on', 'only', 'd',
                             "does",
                             'both', 'once', 'haven', 'during', "don't", 'very', 'yourself', 'be', 'yours', 'where',
                             'him', 'what',
                             "you'll", "would", 'that', 'how', 'ma', 'then', "need", "should", 'there', "that'll", 've',
                             'an', 'out', 'again', 'itself', 'which', 'wouldn', 'any', 'whom', 'above', "you'd", 'just',
                             'for',
                             "has", "were"}


def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"shan't", "refuse to", phrase)
    phrase = re.sub(r"shan", "happy to", phrase)
    phrase = re.sub(r":\(", "bad", phrase)
    phrase = re.sub(r":\)", "good", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def remove_punctuation(text):

    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

    return text.translate(remove_punct_map)


def process_reviews(dirty_data_set):
    clean_data_set = []
    for review in dirty_data_set:

        # Language Detect and Translate
        # Translated Data
        review = decontracted(review)
        # Remove punctuations
        review = remove_punctuation(review)
        # To lowercase
        review = review.lower()
        # Remove stop words & Word Stem
        texts = [ps.stem(word) for word in review.lower().split() if word not in customized_stopwords_list]

        try:
            clean_data_set.append(' '.join(texts))
        except:
            pass
    return clean_data_set


def get_category_sp(category):
    from sklearn.feature_extraction import DictVectorizer
    from collections import Counter

    v = DictVectorizer()
    categories_tidy = [re.sub("\'", '', x.strip("[]")).split(',') for x in category]

    # Memory Error when use full data set!
    categories_sp = v.fit_transform(Counter(f) for f in categories_tidy)

    # Sparse Matrix of the category
    # categories_sp.A
    df = pd.DataFrame(categories_sp.A)
    df.columns = v.get_feature_names()
    return df


def count_upper_word(text):
    words = text.split()
    upper_list = [word for word in words if word.isupper()]
    # Remove single capital letter
    count = len([word for word in upper_list if len(word) > 1])

    return count


"""
Scripts for features generating and Models
"""
#################################################################
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

trainDF["text_length"] = pd.Series([len(i) for i in trainDF.text])
trainDF["num_upper_words"] = pd.Series([count_upper_word(x) for x in trainDF.text])
trainDF["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in trainDF.text])

n_test = testDF.shape[0]

testDF["text_length"] = pd.Series([len(i) for i in testDF.text])
testDF["num_upper_words"] = pd.Series([count_upper_word(x) for x in testDF.text])
testDF["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in testDF.text])

# comment_text = [trainDF.text, testDF.text]
# text = pd.concat(comment_text)
# text = process_reviews(text)

"""
Get TF-IDF from Text
"""

num_feature = 1000000

train_tfVec = CountVectorizer(max_features=num_feature)
final_train_textTF = train_tfVec.fit_transform(process_reviews(trainDF.text))
# Get Vocalbulary for generating sparse matrix
train_features = train_tfVec.get_feature_names()

test_tfVec = CountVectorizer(vocabulary=train_features)
final_test_textTF = test_tfVec.fit_transform(process_reviews(testDF.text))

print(final_train_textTF.shape)
print(final_test_textTF.shape)

"""
Get Sparse Matrix from Categories 
"""
comment_categories = [trainDF.categories, testDF.categories]
categories = pd.concat(comment_categories)

from scipy.sparse import csr_matrix, hstack

final_category = get_category_sp(categories)
# a = csr_matrix(final_category.values)
final_train_category = csr_matrix(final_category.iloc[0:n_train, ].values)
final_test_category = csr_matrix(final_category.iloc[n_train: n_train + n_test, ].values)



"""
Extra Features
"""
trainDF['cityID'] = pd.Categorical(trainDF.city).codes
testDF['cityID'] = pd.Categorical(testDF.city).codes

final_train_extra_features = csr_matrix(trainDF[['cityID',
                                                 'num_upper_words',
                                                 'num_exclamation_mark',
                                                 'text_length']].values)
final_test_extra_features = csr_matrix(testDF[['cityID',
                                               'num_upper_words',
                                               'num_exclamation_mark',
                                               'text_length']].values)

"""
Combine all features to get final Train-Test set
"""
finalX_train2 = hstack([final_train_textTF, final_train_category, final_train_extra_features])
finalX_test2 = hstack([final_test_textTF, final_test_category, final_test_extra_features])


print(finalX_train2.shape)
print(finalX_test2.shape)


def random_forest(finalX_train, finalY_train, finalX_test, n_parallel=1):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=10, n_jobs=n_parallel)
    clf = clf.fit(finalX_train, finalY_train)
    finalY_pred = clf.predict(finalX_test)
    return pd.DataFrame(finalY_pred)


def decision_tree(finalX_train, finalY_train, finalX_test):
    from sklearn import tree

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(finalX_train, finalY_train)
    tree.export_graphviz(clf, out_file='sklearn_tree.dot')
    finalY_pred = clf.predict(finalX_test)
    return pd.DataFrame(finalY_pred)
    # pd.DataFrame(finalY_pred).to_csv('predict_RF.csv', index=False)


def neural_network(finalX_train, finalY_train, finalX_test):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
    clf = clf.fit(finalX_train, finalY_train)
    finalY_pred = clf.predict(finalX_test)
    return pd.DataFrame(finalY_pred)


"""
Fit RF model
"""
# final_RF_pred = random_forest(finalX_train2, trainDF.stars, finalX_test2, n_parallel=4)
# final_RF_pred.to_csv('predict_RF.csv', index=False)

"""
Fit Decision Tree model
"""
# final_DT_pred = decision_tree(finalX_train2, trainDF.stars, finalX_test2)
# final_DT_pred.to_csv('predict_DTree.csv', index=False)

"""
Fit NN model
"""
final_NN_pred = neural_network(finalX_train2, trainDF.stars, finalX_test2)
final_NN_pred.to_csv('predict_NN.csv', index=False)
