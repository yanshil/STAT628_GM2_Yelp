import pandas as pd

import os

path = 'C:\\Users\\kdrob\\Downloads'
os.chdir(path)

train_filename = 'train_data.csv'
test_filename = 'testval_data.csv'

isTrain = False

if isTrain:
    trainDF = pd.read_csv(train_filename)
    text = trainDF.text
else:
    testDF = pd.read_csv(test_filename)
    text = testDF.text

df = pd.DataFrame(columns=['translated', 'lang', 'conf'])

for review in text:
    from googletrans import Translator

    translator = Translator()
    detected = translator.detect(review)
    if detected.lang != 'en':
        translated = translator.translate(review, dest='en')
        df = df.append({"translated": translated.text,
                        "lang": detected.lang,
                        "conf": detected.confidence}
                       , ignore_index=True)
    else:
        df = df.append({"translated": review,
                        "lang": detected.lang,
                        "conf": detected.confidence}
                       , ignore_index=True)

if isTrain:
    df.to_csv("translated_train_text.csv")
else:
    df.to_csv("translated_test_text.csv")

