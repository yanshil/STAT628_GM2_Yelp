"""
GCP Free Trail Credential is not enough to analysis all the data.
Skip this sentiment Analysis Method.
"""
#
# import pandas as pd
#
# # Imports the Google Cloud client library
# from google.cloud import language
# from google.cloud.language import enums
# from google.cloud.language import types
#
#
# # Instantiates a client
# client = language.LanguageServiceClient()
#
# yelp = pd.read_csv('C:\\Users\\kdrob\\Downloads\\train_data.csv')
#
# X = yelp['text']
#
# # test = yelp.head(10)
# #
# # X = test['text']
#
#
# df = pd.DataFrame()
#
# for text in X:
#     document = types.Document(
#         content=text,
#         type=enums.Document.Type.PLAIN_TEXT)
#
#     try:
#         # Detects the sentiment of the text
#         sentiment = client.analyze_sentiment(document=document).document_sentiment
#         df = df.append({'score': sentiment.score, 'magnitude': sentiment.magnitude}, ignore_index=True)
#     except Exception as inst:
#         print(inst)
#         print(text)
#         df = df.append({'score': 6666, 'magnitude': 8888}, ignore_index=True)
#
#     # print('Text: {}'.format(text))
#     # print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))
#
# df.to_csv('GCP_sentiment.csv', index=False)
