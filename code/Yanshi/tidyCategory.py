import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import csv

yelp = pd.read_csv('C:\\Users\\kdrob\\Downloads\\train_data.csv')

tiny = True

if tiny:
    test = yelp.head(100)
    C = test['categories']
else:
    C = yelp['categories']

# Remove "[", "]", "'" and split with comma
C0 = [re.sub("\'", '', x.strip("[]")).split(',') for x in C]

categories_counter = Counter()
for x in C0:
    categories_counter.update(x)

categories_dict = dict(categories_counter)

if tiny:
    with open('categories_dict_tiny.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, categories_dict.keys())
        w.writeheader()
        w.writerow(categories_dict)

    v = DictVectorizer()
    categories_sp = v.fit_transform(Counter(f) for f in C0)  # Memory Error when use full data set!

    # Sparse Matrix of the category
    categories_sp.A
    df = pd.DataFrame(categories_sp.A)
    df.columns = v.get_feature_names()

    df.to_csv('categories_sp_tidy.csv', index=False)

else:
    with open('categories_dict.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, categories_dict.keys())
        w.writeheader()
        w.writerow(categories_dict)