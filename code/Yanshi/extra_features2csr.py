#!/usr/bin/env python3

"""
extra_features2csr.py: Tidy Non-Text/Category features to csr sparse matrix
"""

__author__ = "Yanshi Luo"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"


import pandas as pd
import process_text
import re
from scipy.sparse import csr_matrix


def get_extra_features(data, text_length=True, num_upper_words=True,
                       num_exclamation_mark=True, city=True,
                       num_question_mark=True, num_dollar=True,
                       num_percent=True, num_facebad=True, num_facegood=True):

    data = data.reset_index()
    feature_list = []
    if text_length:
        data["text_length"] = pd.Series([len(i) for i in data.text])
        feature_list.append('text_length')

    if num_upper_words:
        data["num_upper_words"] = pd.Series([process_text.count_upper_word(x) for x in data.text])
        feature_list.append('num_upper_words')

    if num_exclamation_mark:
        data["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in data.text])
        feature_list.append('num_exclamation_mark')

    if city:
        data['cityID'] = pd.Categorical(data.city).codes
        feature_list.append('cityID')

    if num_question_mark:
        data['num_question_mark'] = pd.Series([len(re.findall(r'\?', x)) for x in data.text])
        feature_list.append('num_question_mark')

    if num_dollar:
        data['num_dollar'] = pd.Series([len(re.findall(r'\$', x)) for x in data.text])
        feature_list.append('num_dollar')

    if num_percent:
        data['num_percent'] = pd.Series([len(re.findall(r'\%', x)) for x in data.text])
        feature_list.append('num_percent')

    if num_facebad:
        data['num_facebad'] = pd.Series([len(re.findall(r'\:\(', x)) for x in data.text])
        feature_list.append('num_facebad')

    if num_facegood:
        data['num_facegood'] = pd.Series([len(re.findall(r'\:\)', x)) for x in data.text])
        feature_list.append('num_facegood')

    final_extra_features = csr_matrix(data[feature_list].values)

    return final_extra_features


def label_input(filename):
    data = pd.read_csv(filename)

    return csr_matrix(data)



# comment_categories = [trainDF.categories, testDF.categories]
    # categories = pd.concat(comment_categories)
    #
    # final_category = process_text.get_category_sp(categories)
    # # a = csr_matrix(final_category.values)
    # final_train_category = csr_matrix(final_category.iloc[0:n_train, ].values)
