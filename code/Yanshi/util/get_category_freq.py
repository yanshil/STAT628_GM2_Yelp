#!/usr/bin/env python3

"""
get_category_freq.py: Tidy Yelp Review Categories. Get Frequency Table (and Sparse Matrix)

Example Usage:

python3 get_category_freq.py -s train_output0.csv categories_dict
"""

__author__ = "Yanshi Luo"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"


def get_category_freq(review_filename, output, tiny, sp):
    import pandas as pd
    import re
    from collections import Counter
    import csv

    yelp = pd.read_csv(review_filename)

    if tiny:
        yelp_categories = yelp['categories'].head(100)
    else:
        yelp_categories = yelp['categories']

    # Remove "[", "]", "'" and split with comma
    yelp_categories_tidy = [re.sub("\'", '', x.strip("[]")).split(',') for x in data.categories]

    categories_counter = Counter()
    for x in yelp_categories_tidy:
        categories_counter.update(x)

    categories_dict = dict(categories_counter)

    with open(str(output + '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, categories_dict.keys())
        w.writeheader()
        w.writerow(categories_dict)

    if sp:

        from sklearn.feature_extraction import DictVectorizer

        v = DictVectorizer()

        # Memory Error when use full data set!
        categories_sp = v.fit_transform(Counter(f) for f in yelp_categories_tidy)

        # Sparse Matrix of the category
        # categories_sp.A
        df = pd.DataFrame(categories_sp.A)
        df.columns = v.get_feature_names()

        df.to_csv(str(output + '_sp.csv'), index=False)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Tidy Yelp Review Categories. Get Frequency Table (and Sparse Matrix)')
    parser.add_argument('review_filename',
                        help='The csv file you\'d like to analysis.')
    parser.add_argument('output',
                        help='The prefix of the output csv you\'d like to stored as.')
    parser.add_argument('-t', '--tiny',
                        help='Use a smaller data set to test the code.',
                        default=False,
                        action='store_true')
    parser.add_argument('-s', '--sparseMatrix',
                        help='Get Sparse Matrix output.',
                        default=False,
                        action='store_true')
    args = parser.parse_args()

    get_category_freq(args.review_filename, args.output, args.tiny, args.sparseMatrix)
