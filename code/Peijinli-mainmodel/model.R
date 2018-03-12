setwd('C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2')
library('hydroGOF')
library(nnet)
library('readr')
train <- read.csv("m2_train.csv")
test <- read.csv("m2_test.csv")
finaltest<-read.csv("finaltest.csv")

############################################# combine the significant city and name ############################################

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

library(tidyverse)
library(dplyr)

city_bt<-read_csv("adjustment on area.csv")
name_bt<-read_csv("byrestaurant.csv")

f_city <- city_bt %>% filter(adjustment > 0) %>% select(city) %>% data.frame()
f_name <- name_bt %>% filter(adjustment > 1) %>% select(name) %>% data.frame()
f_city$city_idx <- c(1:dim(f_city)[1])
f_name$name_idx <- c(1:dim(f_name)[1])

train <- train %>% left_join(f_city,by = 'city')
train <- train %>% left_join(f_name,by = 'name')
test <- test %>% left_join(f_city,by = 'city')
test <- test %>% left_join(f_name,by = 'name')
finaltest <- finaltest %>% left_join(f_city,by = 'city')
finaltest <- finaltest %>% left_join(f_name,by = 'name')

train$city_idx[is.na(train$city_idx)]<-0
train$name_idx[is.na(train$name_idx)]<-0
test$city_idx[is.na(test$city_idx)]<-0
test$name_idx[is.na(test$name_idx)]<-0
finaltest$city_idx[is.na(finaltest$city_idx)]<-0
finaltest$name_idx[is.na(finaltest$name_idx)]<-0

################################################### based on the simple linear regression model ###############################

city_sig <- c(5,7,8,11,13,14,17,18,20,23,26,29,30,31,32,35,39,46,47,48,49,50,54,57,62,63,66)
name_sig <-c(1,2,3,5,7,10,11,13,16,17,19,20,26,27,28,34,37,41,44,45,46,47,50,52,53,58,59,
                 62,63,65,67,69,72,73,75,76,77,78,83,84,86,87,88,90:94,101:103,107,109,110)

#city1 <- f_city %>% filter(city_idx %in% city_sig)
#write.csv(city1,'sigcity.csv')
#name1 <- f_name %>% filter(name_idx %in% name_sig)
#write.csv(name1,'signame.csv')

train$city_idx[!train$city_idx %in% city_sig]<-0
test$city_idx[!test$city_idx %in% city_sig]<-0
finaltest$city_idx[!finaltest$city_idx %in% city_sig]<-0


train$name_idx[!train$name_idx %in% name_sig]<-0
test$name_idx[!test$name_idx %in% name_sig]<-0
finaltest$name_idx[!finaltest$name_idx %in% name_sig]<-0

train$city_idx<-as.factor(train$city_idx)
train$name_idx<-as.factor(train$name_idx)
test$city_idx<-as.factor(test$city_idx)
test$name_idx<-as.factor(test$name_idx)
finaltest$city_idx<-as.factor(finaltest$city_idx)
finaltest$name_idx<-as.factor(finaltest$name_idx)

length(unique(train$city_idx))
length(unique(train$name_idx))
length(unique(test$city_idx))
length(unique(test$name_idx))
length(unique(finaltest$city_idx))
length(unique(finaltest$name_idx))

str(train)
str(test)
str(finaltest)

train[450000,]
test[150000,]
finaltest[1016664,]

############################################# fit linear regression ##################################################


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


