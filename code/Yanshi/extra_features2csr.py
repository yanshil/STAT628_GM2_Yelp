import pandas as pd
import process_text
import re
from scipy.sparse import csr_matrix


def get_extra_features(data, text_length=True,
                       num_upper_words=False, num_exclamation_mark=False,
                       city=False):

    # n_train = trainDF.shape[0]

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

    final_extra_features = csr_matrix(data[feature_list].values)

    return final_extra_features


# comment_categories = [trainDF.categories, testDF.categories]
    # categories = pd.concat(comment_categories)
    #
    # final_category = process_text.get_category_sp(categories)
    # # a = csr_matrix(final_category.values)
    # final_train_category = csr_matrix(final_category.iloc[0:n_train, ].values)
