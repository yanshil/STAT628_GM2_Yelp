"""
Grasp Info From Data
"""
import pandas as pd
import re
import numpy as np

#################################################################
import os
path = 'C:\\Users\\kdrob\\Downloads'
os.chdir(path)

train_filename = 'train_data.csv'
test_filename = 'testval_data.csv'

trainDF = pd.read_csv(train_filename)


def count_upper_word(text):
    words = text.split()
    upper_list = [word for word in words if word.isupper()]
    # Remove single capital letter
    count = len([word for word in upper_list if len(word) > 1])

    return count


trainDF["num_upper_words"] = pd.Series([count_upper_word(x) for x in trainDF.text])
trainDF["num_exclamation_mark"] = pd.Series([len(re.findall(r'!', x)) for x in trainDF.text])

"""
num_exclamation_mark & Upper words
"""

trainDF.groupby(['stars', 'num_exclamation_mark']).size().to_frame().reset_index().to_csv('num_exclamation_mark.csv',
                                                                                          index=False)
trainDF.groupby(['stars', 'num_upper_words']).size().to_frame().reset_index().to_csv('num_upper_words.csv',
                                                                                     index=False)

trainDF.groupby('stars').size().reset_index().to_csv('num_stars_comments.csv', index = False)


np.where(trainDF["num_exclamation_mark"] == 44)
#
# trainDF.text[780900, ]
# "the worst meal in montreal!!!! the french onion soup was a joke!!!!! the crepes were awful!!! do not go to this restaurant under any circumstance unless you want to throw away money and time in your life you can't get back...THE OWNER AND CHEF IS RUDE AND CAN'T COOK!!!!!!!!  i wish crepe bretonne was open next door but it closed and this JOKE OF A RESTUARANT ADVERTISES THEY ARE THE SAME!!!!!\nLIES,LIES AND LIES!!!!!!\nI HATED MY MEAL!!!!!!!!!!!!!"
# np.where(trainDF["num_exclamation_mark"] == 44)
# (array([  26856,   63121,  230525,  315084,  367773,  496274,  513270,
#         780900,  811868,  915783, 1098242, 1125951, 1445171, 1469737],
#       dtype=int32),)

trainDF.text[63121, ]
