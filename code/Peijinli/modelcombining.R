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


      
####################################################################################################
#  question:0.5276  #0.5285411 #^2  0.5267685 # ^3 0.526038 #^4  0.5255427                         #
#  10 variables 0.5166709 #10var ^2 0.5166698 #12 0.516705  #add catscore 0.5160276                #
####################################################################################################

m1<-lm(stars~name_idx+city_idx+X0+catscore+text_length+num_upper_words+num_exclamation_mark+question_mark+dollar+precent+facebad+facegood+goodcat+badcat+rest_length+
       I(X0^2)+I(catscore^2)+I(text_length^2)+I(num_exclamation_mark^2)+I(question_mark^2)+I(dollar^2)+
         I(X0^3)+I(text_length^3)+I(num_exclamation_mark^3)+I(question_mark^3)+
         I(X0^4)+I(text_length^4)+I(num_exclamation_mark^4)+I(question_mark^4)
         ,data=train)

summary(m1) #0.6845
#0.6839city and name
t1<-predict(m1,test)
mse(t1,test$stars)   #0.5157375 : city # 0.5156141 :name #0.5153286 :name+city 
mse(test$X0,test$stars)

t2<-predict(m1,finaltest)
write.csv(t2,'final_pred.csv')

write.csv(a,'final_pred55.csv')
################################################  nnet ###############################

m1<-lm(stars~text_length+num_upper_words+num_exclamation_mark+question_mark+
         dollar+precent+facebad+facegood+goodcat+badcat+rest_length
       ,data=train)
summary(m1) #0.6845

################################################ model2 ##############################################
#14var 13 remain 
drops <- c('X','city','name','X0','catscore')
train1<- train[ ,!(names(train)) %in% drops]
str(train1)

m2<-lm(stars~.,data = train1)
summary(m2)
t2_1<-predict(m2,test)
mse(t2_1,test$stars)#0.5264 except city name idx #include city:0.5262297 #include sigcity0.526234
mse(test$X0,test$stars)

t2_2<-predict(m2,finaltest)
write.csv(t2_2,'final_pred.csv')
###############################################decision tree#################################
library(rpart)
fit <- rpart(stars~.,method="class",data = train)
printcp(fit) # display the results #0.63
#plotcp(fit) # visualize cross-validation results 
#summary(fit) # detailed summary of splits

fit1 <- predict(fit,test,type = 'class')
fit_1<-data.frame(fit1)
mse(as.numeric(fit_1$fit1),test$stars)#0.5259876
############################################## svm###########################################
library('e1071')
model <- svm(stars ~ ., data = train)

############################################## nn############################################

library(Rcpp)
library(RSNNS)
cvdat<-rbind(train,test)
p<-as.matrix(cvdat)
###################################################



res<-read.csv('final_pred.csv')
res$Prediction1
result <- c()
for (i in 1:length(res$Id)){
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


