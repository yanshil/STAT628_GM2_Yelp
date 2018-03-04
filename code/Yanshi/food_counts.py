
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

with open('./index/drink.txt', 'r') as f:
    drink_list = [line.strip() for line in f]

with open('./index/fruit.txt', 'r') as f:
    fruit_list = [line.strip() for line in f]

with open('./index/meat.txt', 'r') as f:
    meat_list = [line.strip() for line in f]

with open('./index/milk product.txt', 'r') as f:
    milk_list = [line.strip() for line in f]

with open('./index/nuts.txt', 'r') as f:
    nuts_list = [line.strip() for line in f]

with open('./index/oil.txt', 'r') as f:
    oil_list = [line.strip() for line in f]

with open('./index/vegetable.txt', 'r') as f:
    vegetable_list = [line.strip() for line in f]


def count_foods(text, food_list):
    words = text.lower().split()
    # Remove single capital letter
    count_list = [word for word in words if word in food_list]
    count = len([word for word in count_list])

    return count


########################3
import os

path = 'C:\\Users\\kdrob\\Downloads'
os.chdir(path)

train_filename = 'train_data.csv'
test_filename = 'testval_data.csv'


full_list = drink_list + fruit_list + meat_list + milk_list + nuts_list + oil_list + vegetable_list
full_list = [word.lower() for word in full_list]


train_filename = 'train_data.csv'

trainDF = pd.read_csv(train_filename)

food_counter = CountVectorizer(vocabulary=full_list)
food_count = food_counter.fit_transform(trainDF.text)
food_name = food_counter.get_feature_names()

final_food_count = pd.DataFrame(food_count.toarray(), columns=food_name)
final_food_count["stars"] = trainDF["stars"]

final_food_count.groupby(['stars']).sum().reset_index().to_csv('final_food_count.csv', index=False)

# trainDF["drink_count"] = pd.Series([count_foods(x, drink_list) for x in trainDF.text])
# trainDF.groupby(['stars', 'drink_count']).size().to_frame().reset_index().to_csv('num_drink_count.csv',
#                                                                                  index=False)
