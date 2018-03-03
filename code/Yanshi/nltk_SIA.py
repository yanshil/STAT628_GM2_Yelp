#!/usr/bin/env python3

"""nltk_SIA.py: Sentiment Intensity Analysis on Yelp Review Text with NLTK."""

__author__ = "Yanshi Luo"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"


def process(yelp_filename, output_filename, tiny):
    import pandas as pd
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import csv

    """Run a sentiment analysis request on text within a passed filename."""
    yelp = pd.read_csv(yelp_filename)

    if tiny:
        yelp_comments = yelp['text'].head(100)
    else:
        yelp_comments = yelp['text']

    sid = SentimentIntensityAnalyzer()

    first_line = sid.polarity_scores(yelp_comments[1])

    with open(output_filename, 'w', newline='') as f:
        # Write Headers
        writer = csv.DictWriter(f, first_line.keys())
        writer.writeheader()

        for sentence in yelp_comments:
            ss = sid.polarity_scores(sentence)
            writer.writerow(ss)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run Sentiment Analysis on Yelp Data.')
    parser.add_argument('review_filename',
                        help='The filename of the yelp review you\'d like to analyze.')
    parser.add_argument('output_filename',
                        help='The filename of the output you\'d like to stored as.')
    parser.add_argument('-t', '--tiny',
                        help='Use a smaller data set to test the code.',
                        default=False,
                        action='store_true')
    args = parser.parse_args()

    process(args.review_filename, args.output_filename, args.tiny)

