#!/usr/bin/env python3

"""
process_text.py: Define Text processing rules
"""

__author__ = "Yanshi Luo", "Peijin"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"

import re


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
    import string

    remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

    return text.translate(remove_punct_map)


def process_reviews(dirty_data_set):

    customized_stopwords_list = {'ll', 'are', 'theirs', 'up', 'do', 'have', 'who', 'few', 'needn', 'yourselves', 'has',
                                 'under', 'ain', 'the', 'should', 'y', 'might', "must", 'was', 'had', 'she', 'is',
                                 'through',
                                 'himself', 'their', 'ours', 'm', 'and', 'am', 'against', 'his', 'from', 'mustn', 'off',
                                 'her',
                                 "will", 'myself', 'as', "did", "is", 'themselves', 'o', 'of', 'them', 'does', 'i', 'a',
                                 'by',
                                 't', 'had', 'it', 'after', "should've", 'was', 'did', 'my', 'into', 'they', 'such',
                                 'but',
                                 'if',
                                 'hers', 'with', 'your', 'than', "had", 'did', "could", "she's",
                                 "might", 'has', 'each', 'these', 'our', 'will', 'those', 'can', 'he', 'over', 'could',
                                 'having', 'below', 'between', 'own', 'until', 'about', 'all', 'being', 'why', 'should',
                                 'most', 're', 'we', 'doing', 'at', 'because', 's', 'does', 'now', 'other', 'down',
                                 'ourselves', 'so', 'you', 'were', 'while', 'to', 'here', 'me', "you've", 'its',
                                 'herself',
                                 'further', 'too', 'isn', "you're", 'were', "was", 'some', 'in', 'been', "it's", 'or',
                                 'are', 'nor', "have", 'same', 'before', 'won', 'when', 'more', 'this', 'on', 'only',
                                 'd',
                                 "does",
                                 'both', 'once', 'haven', 'during', "don't", 'very', 'yourself', 'be', 'yours', 'where',
                                 'him', 'what',
                                 "you'll", "would", 'that', 'how', 'ma', 'then', "need", "should", 'there', "that'll",
                                 've',
                                 'an', 'out', 'again', 'itself', 'which', 'wouldn', 'any', 'whom', 'above', "you'd",
                                 'just',
                                 'for',
                                 "has", "were"}

    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    clean_data_set = []
    for review in dirty_data_set:

        # Language Detect and Translate
        # Translated Data
        review = decontracted(review)
        # Remove punctuations
        review = remove_punctuation(review)
        # Remove blanks
        review = re.sub(r'\s+', ' ', review)
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
    import pandas as pd
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

