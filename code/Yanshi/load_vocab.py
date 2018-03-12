import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import process_text

train_filename = 'train_data.csv'
# test_filename = 'testval_data.csv'

trainDF = pd.read_csv(train_filename)
# testDF = pd.read_csv(test_filename)

good_cate_list = ['Delis', 'Polish', 'Vegan', 'French', 'Peruvian']
bad_cate_list = ['Chicken_Wings', 'Fast_Food', 'Buffets', 'Tex-Mex', 'Burgers']
full_list = [x.lower() for x in good_cate_list + bad_cate_list]


def get_category_features_sp(category, vocab):

    categories2list = [re.sub("\'", '', x.strip("[]")).split(',') for x in category]

    str_cat = []
    for cate in categories2list:
        record = [re.sub(' ', '_', x.strip()) for x in cate]
        str_cat.append(' '.join(record))

    vocab = [x.lower() for x in vocab]

    category_counter = CountVectorizer(vocabulary=vocab)
    category_sp = category_counter.fit_transform(str_cat)
    return category_sp


def get_text_features_sp(text, vocab):
    text = process_text(text)
    # Unique
    vocab = list(set(vocab))
    text_feature_counter = CountVectorizer(vocabulary=vocab)
    text_feature_sp = text_feature_counter.fit_transform(text)

    return text_feature_sp


def read_features2vocab(filename):
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()
    with open(filename, 'r', encoding='utf8') as f:
        feature_list = [line.decode('utf-8').strip() for line in f]

    return [ps.stem(word).lower() for word in feature_list]



