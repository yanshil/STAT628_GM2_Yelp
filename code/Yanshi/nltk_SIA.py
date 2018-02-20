import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv

yelp = pd.read_csv('C:\\Users\\kdrob\\Downloads\\train_data.csv')

# test = yelp.head(100)
#
# X = test['text']

X = yelp['text']

sid = SentimentIntensityAnalyzer()

s0 = sid.polarity_scores(X[1])

with open('sentimentScore.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, s0.keys())
    w.writeheader()
    for sentence in X:
        ss = sid.polarity_scores(sentence)
        w.writerow(ss)

# for sentence in X:
#     print(sentence)
#     ss = sid.polarity_scores(sentence)
#     for k in sorted(ss):
#         print('{0}: {1}, '.format(k, ss[k]), end='')
#     print()
