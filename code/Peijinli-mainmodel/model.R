setwd('C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2')
library('hydroGOF')
library(nnet)
library('readr')
train <- read.csv("m2_train.csv")
test <- read.csv("m2_test.csv")
finaltest<-read.csv("finaltest.csv")

train2<-read_csv("train_data.csv")
test2<-read_csv("testval_data.csv")
str(train2)
str(test)

train$city <- train2[1:450000,]$city
test$city <- train2[450001:600000,]$city 
finaltest$city <- test2$city

train$name <- train2[1:450000,]$name
test$name <- train2[450001:600000,]$name
finaltest$name <- test2$name


m1<-lm(stars~name_idx+city_idx+X0+catscore+text_length+num_upper_words+num_exclamation_mark+question_mark+dollar+precent+facebad+facegood+goodcat+badcat+rest_length+
       I(X0^2)+I(catscore^2)+I(text_length^2)+I(num_exclamation_mark^2)+I(question_mark^2)+I(dollar^2)+
         I(X0^3)+I(text_length^3)+I(num_exclamation_mark^3)+I(question_mark^3)+
         I(X0^4)+I(text_length^4)+I(num_exclamation_mark^4)+I(question_mark^4)
         ,data=train)

summary(m1)
t1<-predict(m1,test)
mse(t1,test$stars)  
t2<-predict(m1,finaltest)
for (i in 1:length(t2)){
  if(res$Prediction1[i]<1){
    a = 1
  }else if (res$Prediction1[i]>5){
    a = 5
  }else{
    a = res$Prediction1[i]
  }
  result<-c(result,a)
}
write.csv(result,'final_pred2.csv')


