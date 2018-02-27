# STAT 628: Mon Group 2

2018/2/21 10:38
Update: Test Set Sentiment Score is generated. See [sentimentScore_test.csv](./data/Yanshi/sentimentScore_test.csv) 

### Description

* `./code/Yanshi/nltk_SIA.py` Generate sentiment Analyze Score from [nltk.sentiment.sentiment_analyzer module](https://www.nltk.org/api/nltk.sentiment.html?highlight=sentiment#module-nltk.sentiment.sentiment_analyzer). 
  * Training Set Output: `./data/Yanshi/sentimentScore.csv`.
  * Test Set Output: `./data/Yanshi/sentimentScore_test.csv`.
  * Format: `compound: 0.8316, neg: 0.0, neu: 0.254, pos: 0.746`
  * How-to:  [Sentiment](http://www.nltk.org/howto/sentiment.html)



分工：
netural network: shi

random forest: shi

dicision tree: shi

Bayesian:both

xgb: both

adboost: both

LDA: pei

logistic: pei

SVM: pei

罗决策树:pei

黄文：

data cleaning: stop words, 正则表达式, TFIDF ,划词 

model1：nltk + 餐厅名字 

餐厅地图表 + 圆圈:数量 + 颜色：avg score (红绿色)

base line

已经试过的模型及其结果

未来要做的

拟合新变量!!!! foodname drinkname