import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from nltk.stem import PorterStemmer
import numpy as np

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


def process_reviews(dirty_data_set):
    clean_data_set = []
    for review in dirty_data_set:
        review = decontracted(review)
        # Remove punctuations
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
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


#################################################################
# import os
# path = 'C:\\Users\\kdrob\\Downloads'
# os.chdir(path)

train_filename = 'train_data.csv'
test_filename = 'testval_data.csv'
# dt_train = pd.read_csv(train_filename)
# dt_test = pd.read_csv(test_filename)

trainDF = pd.read_csv(train_filename)
testDF = pd.read_csv(test_filename)

# dt_train = pd.read_csv(train_filename, sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],
#                        engine='python')
#
# dt_test = pd.read_csv(test_filename, sep=",", error_bad_lines=False, keep_default_na=False, na_values=[""],
#                       engine='python')

#trainDF = dt_train
# trainDF = dt_train.head(100)
n_train = trainDF.shape[0]
# trainDF = range(trainDF.shape[0])
#trainDF.text = process_reviews(trainDF.text)
trainDF["text_length"] = pd.Series([len(i) for i in trainDF.text])
trainDF["num_upper_words"] = pd.Series([count_upper_word(x) for x in trainDF.text])
trainDF["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in trainDF.text])

#testDF = dt_test
# testDF = dt_test.head(2)
n_test = testDF.shape[0]
# testDF = range(testDF.shape[0])
#testDF.text = process_reviews(testDF.text)
testDF["text_length"] = pd.Series([len(i) for i in testDF.text])
testDF["num_upper_words"] = pd.Series([count_upper_word(x) for x in testDF.text])
testDF["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in testDF.text])


comment_text = [trainDF.text, testDF.text]
text = pd.concat(comment_text)
text = process_reviews(text)
# text.index = range(text.shape[0])

comment_categories = [trainDF.categories, testDF.categories]
categories = pd.concat(comment_categories)
# cate.index = range(cate.shape[0])

num_feature = 1000000


train_tfVec = TfidfVectorizer(max_features=num_feature)
final_train_textTF = train_tfVec.fit_transform(trainDF.text)
# Get Vocalbulary for generating sparse matrix
train_features = train_tfVec.get_feature_names()

test_tfVec = TfidfVectorizer(vocabulary=train_features)
final_test_textTF = test_tfVec.fit_transform(trainDF.text)

# # tf_vectorizer = CountVectorizer(max_features=num_feature)
# final_textTF = tf_vectorizer.fit_transform(text)
# tf_vectorizer.get_feature_names()

print(n_train)
print(n_test)
# print(final_textTF.shape)

################### split the train and test to do the model fitting####################
#
# final_train_textTF = final_textTF[range(0, n_train), ]
# final_test_textTF = final_textTF[range(n_train, n_train + n_test), ]

final_category = get_category_sp(categories)
final_train_category = final_category.iloc[0:n_train, ]
final_test_category = final_category.iloc[n_train: n_train + n_test, ]
# final_train_category = final_category[range(0, n_train), ]
# final_test_category = final_category[range(n_train, n_train + n_test), ]

#################### combine other feature ############################################
finalX_train2 = np.hstack((final_train_textTF.toarray(),
                           np.array(trainDF['longitude'])[:, None],
                           np.array(trainDF['latitude'])[:, None],
                           final_train_category,
                           np.array(trainDF.num_upper_words)[:, None],
                           np.array(trainDF.num_exclamation_mark)[:, None],
                           np.array(trainDF.text_length)[:, None]))

finalX_test2 = np.hstack((final_test_textTF.toarray(),
                          np.array(testDF['longitude'])[:, None],
                          np.array(testDF['latitude'])[:, None],
                          final_test_category,
                          np.array(testDF.num_upper_words)[:, None],
                          np.array(testDF.num_exclamation_mark)[:, None],
                          np.array(testDF.text_length)[:, None]))

# ################### fitting svm model #################################################
# weight = getproportion(finalX_train2,'stars')
# wclf = svm.SVC(kernel='linear', class_weight=weight)
# wclf.fit(X_train2, trainDF.stars)
# finalpredY = wclf.predict(finalX_test2)
# ##################print output#########################################################

from sklearn.ensemble import RandomForestClassifier

# clf = RandomForestClassifier(n_estimators=10, n_jobs=4)
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(finalX_train2, trainDF.stars)
final_predY = clf.predict(finalX_test2)
pd.DataFrame(final_predY).to_csv('predict_RF.csv', index=False)

##########################
# Decision Tree
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(finalX_train2, trainDF)
# final_predY = clf.predict(finalX_test2)
# pd.DataFrame(final_predY).to_csv('predict.csv', index=True)

