# STAT 628: Mon Group 2

2018/2/21 10:38
Update: Test Set Sentiment Score is generated. See [sentimentScore_test.csv](./data/Yanshi/sentimentScore_test.csv) 

### Description

* `./code/Yanshi/nltk_SIA.py` Generate sentiment Analyze Score from [nltk.sentiment.sentiment_analyzer module](https://www.nltk.org/api/nltk.sentiment.html?highlight=sentiment#module-nltk.sentiment.sentiment_analyzer). 
  * Training Set Output: `./data/Yanshi/sentimentScore.csv`.
  * Test Set Output: `./data/Yanshi/sentimentScore_test.csv`.
  * Format: `compound: 0.8316, neg: 0.0, neu: 0.254, pos: 0.746`
  * How-to:  [Sentiment](http://www.nltk.org/howto/sentiment.html)