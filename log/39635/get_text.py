import pandas as pd

# import os
# path = 'C:\\Users\\kdrob\\Downloads'
# os.chdir(path)

train_filename = 'train_data.csv'
test_filename = 'testval_data.csv'

trainDF = pd.read_csv(train_filename)
testDF = pd.read_csv(test_filename)

train_text = trainDF.text
test_text = trainDF.text

train_text.to_csv("train_text.csv", index=None)
test_text.to_csv("test_text.csv", index=None)


