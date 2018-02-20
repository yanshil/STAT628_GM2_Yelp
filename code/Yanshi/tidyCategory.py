import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction import DictVectorizer

yelp = pd.read_csv('C:\\Users\\kdrob\\Downloads\\train_data.csv')

# C = yelp['categories']

test = yelp.head(100)

C = test['categories']

# Remove "[", "]", "'" and split with comma
C0 = [re.sub("\'", '', x.strip("[]")).split(',') for x in C]

categories_counter = Counter()
for x in C0:
    categories_counter.update(x)

categories_dict = dict(categories_counter)

v = DictVectorizer()
categories_sp = v.fit_transform(Counter(f) for f in C0)  # Memory Error when use full data set!

# Sparse Matrix of the category
categories_sp.A
df = pd.DataFrame(categories_sp.A)
df.columns = v.get_feature_names()

df.to_csv('categories_sp_tidy.csv', index=False)
