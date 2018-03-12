setwd('C:\\Users\\Peiji\\Desktop\\spring2018\\STAT628\\hw2')
library('hydroGOF')
library("readr")
library("ggplot2")
train <- read.csv("m2_train.csv")
test <- read.csv("m2_test.csv")
finaltest<-read.csv("finaltest.csv")

####################################################################################################
#  question:0.5276  #0.5285411 #^2  0.5267685 # ^3 0.526038 #^4  0.5255427                         #
#  10 variables 0.5166709 #10var ^2 0.5166698 #12 0.516705  #add catscore 0.5160276                #
####################################################################################################

m1<-lm(stars~X0+catscore+text_length+num_upper_words+num_exclamation_mark+question_mark+dollar+precent+facebad+facegood+goodcat+badcat+rest_length+
       I(X0^2)+I(catscore^2)+I(text_length^2)+I(num_exclamation_mark^2)+I(question_mark^2)+I(dollar^2)+I(badcat^2)+
         I(X0^3)+I(catscore^3)+I(text_length^3)+I(num_exclamation_mark^3)+I(question_mark^3)+
         I(X0^4)+I(catscore^4)+I(text_length^4)+I(num_exclamation_mark^4)+I(question_mark^4)
         ,data=train)

summary(m1) #0.6845

t1<-predict(m1,test)
mse(t1,test$stars)

mse(test$X0,test$stars)

t2<-predict(m1,finaltest)
write.csv(t2,'final_pred.csv')


## Model Diagnoses
sample = sample(m1$residuals,500,replace = T)
sample = data.frame(y = sample)
index = c(1:dim(sample)[1])
residuals = sample$y
ggplot(data=sample,aes(x = index, y = residuals)) + 
  geom_point(color = "darkred") + geom_smooth(color = "blue",method = 'loess') 

################################################ model2 ##############################################
#14var 13 remain 

drops <- c('X')

train <- train[,!(names(train) %in% drops)]
test <- test[,!(names(test) %in% drops)]
finaltest <- finaltest[,!(names(finaltest) %in% drops)]

str(train)
str(test)
str(finaltest)


m2<-lm(stars~.,data = train)
summary(m2)
t2_1<-predict(m2,test)
mse(t2_1,test$stars)#0.5264
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
