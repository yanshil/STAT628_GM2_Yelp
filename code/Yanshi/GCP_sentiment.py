"""Demonstrates how to make a simple call to the Natural Language API."""

import pandas as pd

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

# Instantiates a client
client = language.LanguageServiceClient()

yelp = pd.read_csv('C:\\Users\\kdrob\\Downloads\\train_data.csv')

# X = yelp['text']

test = yelp.head(10)

X = test['text']


df = pd.DataFrame()

for text in X:
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    try:
        # Detects the sentiment of the text
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        df = df.append({'score': sentiment.score, 'magnitude': sentiment.magnitude}, ignore_index=True)
    except Exception as inst:
        print(inst)
        print(text)

    # print('Text: {}'.format(text))
    # print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))

df.to_csv('GCP_sentiment.csv', index=False)

#
# C:\Users\kdrob\AppData\Local\Programs\Python\Python36-32\python.exe "C:/Users/kdrob/OneDrive - UW-Madison/WiscTextbook/Spring 2018/STAT628/Module2/STAT628_GM2_Yelp/code/Yanshi/GCP_sentiment.py"
# Traceback (most recent call last):
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\google\api_core\grpc_helpers.py", line 54, in error_remapped_callable
#     return callable_(*args, **kwargs)
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\grpc\_channel.py", line 487, in __call__
#     return _end_unary_response_blocking(state, call, False, deadline)
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\grpc\_channel.py", line 437, in _end_unary_response_blocking
#     raise _Rendezvous(state, None, None, deadline)
# grpc._channel._Rendezvous: <_Rendezvous of RPC that terminated with (StatusCode.INVALID_ARGUMENT, The language sv is not supported for document_sentiment analysis.)>
#
# The above exception was the direct cause of the following exception:
#
# Traceback (most recent call last):
#   File "C:/Users/kdrob/OneDrive - UW-Madison/WiscTextbook/Spring 2018/STAT628/Module2/STAT628_GM2_Yelp/code/Yanshi/GCP_sentiment.py", line 30, in <module>
#     sentiment = client.analyze_sentiment(document=document).document_sentiment
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\google\cloud\language_v1\gapic\language_service_client.py", line 180, in analyze_sentiment
#     return self._analyze_sentiment(request, retry=retry, timeout=timeout)
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\google\api_core\gapic_v1\method.py", line 139, in __call__
#     return wrapped_func(*args, **kwargs)
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\google\api_core\retry.py", line 260, in retry_wrapped_func
#     on_error=on_error,
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\google\api_core\retry.py", line 177, in retry_target
#     return target()
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\google\api_core\timeout.py", line 206, in func_with_timeout
#     return func(*args, **kwargs)
#   File "C:\Users\kdrob\AppData\Roaming\Python\Python36\site-packages\google\api_core\grpc_helpers.py", line 56, in error_remapped_callable
#     six.raise_from(exceptions.from_grpc_error(exc), exc)
#   File "<string>", line 3, in raise_from
# google.api_core.exceptions.InvalidArgument: 400 The language sv is not supported for document_sentiment analysis.

