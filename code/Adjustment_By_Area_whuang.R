library(MASS)
library(readr)
library(tidyverse)
install.packages("tidyverse")
train_data <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/train_data.csv")
sentimentScore <- read.csv("C:/Users/Wen/Desktop/STAT 628/module2/sentimentScore.csv")
testset <- read.csv("C:/Users/Wen/Desktop/STAT 628/module2/testval_data.csv")
train_data$stars

## Ordered logistic regression

yelp <-  cbind(train_data,sentimentScore)
yelp$stars <-  as.factor(yelp$stars)
yelp.plr <- polr(stars ~ compound, data = yelp)
Summary(yelp.plr,digits = 3)

testval_data <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/testval_data.csv")

ERROR <- as.numeric(predict(yelp.plr)) - as.numeric(yelp$stars)
sum(abs(ERROR))

model2 = lm(as.numeric(stars) ~ compound, data = yelp)
ERROR2 <- as.numeric(round(predict(model2),0)) - as.numeric(yelp$stars)

predict <- predict(model2)
sum(abs(ERROR2))

ERROR3 <- as.numeric(predict(model2),0) - as.numeric(yelp$stars)

predictstar = c()
yelp$stars <- as.numeric(yelp$stars)
for (i in 1:1546379) {
 if(yelp$compound[i] <= -0.6){
   predictstar[i] = 1
 } else if(yelp$compound[i] <= -0.2){
   predictstar[i] = 2
 } else if(yelp$compound[i] <= 0.2){
   predictstar[i] = 3
 } else if(yelp$compound[i] <= 0.6){
   predictstar[i] = 4
 } else{
   predictstar[i] = 5
 }
   
}

sum(abs(predictstar - yelp$stars))
model2 = lm(as.numeric(stars) ~ compound, data = yelp)

## 3 variables

yelp.plr2 <- polr(stars ~ neg+neu+pos, data = yelp)

ERROR3 <- as.numeric(predict(yelp.plr2)) - as.numeric(yelp$stars)
sum(abs(ERROR3))

# linear models
model4 =  lm(as.numeric(stars) ~ neg+neu+pos, data = yelp)
summary(model4)

ERROR4 <- round(predict(model4),0) - as.numeric(yelp$stars)
sum(abs(ERROR4))




SUB <- train_data[1:100000,]
categories <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/categories_dict.csv", col_names = FALSE)
categories <- categories[, -3]  #without restaurants
namelist = c()
for (i in 1:922){
  namelist[i] <- as.character(categories[1,i])
}


average <- mean(SUB$stars)


by_area <- group_by(train_data,city) 
s = summarise(by_area, count =n(), meanstar = mean(stars, na.rm = TRUE))
sss = filter(s,count>30)  # Cities which have # of reviews larher than 30


## Bootstrap
B = 1000
alpha = 0.05
lower = c()
upper = c()
judge = c()
adjustment = c()
for(i in 1:dim(sss)[1]){
  n = sss$count[i]
  set = filter(by_area, city == sss$city[i])                                       
  boot.samples = matrix(sample(set$stars, size = B * n, replace = TRUE), B, n)
  boot.statistics = apply(boot.samples, 1, mean)
  #se = sd(boot.statistics) 
  #me = qnorm(1-alpha/2)*se/sqrt(n) 
  lower[i] = quantile(boot.statistics,alpha)
  upper[i] = quantile(boot.statistics,1-alpha)
  if((3.7<lower[i]) || (3.7>upper[i])){
    judge[i] = "Distant"
    adjustment[i] = round(sss$meanstar[i] - 3.7,3)
  }else{ judge[i] = "Close"
         adjustment[i] = round(0,1) }
}
sss$lower = lower
sss$upper = upper
sss$judge = judge
sss$adjustment = adjustment

