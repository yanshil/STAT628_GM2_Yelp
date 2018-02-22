#!/usr/bin/env python3

"""csv_splitter.py: Split the input csv with Pandas.DataFrame"""

__author__ = "Yanshi Luo"
__license__ = "GPL"
__email__ = "yluo82@wisc.edu"


def splitter(input_filename, output_template, max_rows):
    import pandas as pd

    data = pd.read_csv(input_filename)

    data_frames = []

    print("Expected Number of Output Files: " + str(len(data)//max_rows+1))

    while len(data) > max_rows:
        top = data[:max_rows]
        data_frames.append(top)
        data = data[max_rows:]
    else:
        data_frames.append(data)

    for _, frame in enumerate(data_frames):
        frame.to_csv(output_template + str(_) + '.csv', index=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process Input/Output File')
    parser.add_argument('review_filename',
                        help='The csv file you\'d like to spli.')
    parser.add_argument('output_template',
                        help='The prefix of the output csv you\'d like to stored as.')
    parser.add_argument('-r', "--rows",
                        help='maximum rows of a splitted csv file.',
                        type=int,
                        default=4500)
    args = parser.parse_args()

    splitter(args.review_filename, args.output_template, args.rows)
