library(MASS)
library(readr)
library(tidyverse)
train_data <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/train_data.csv")
sentimentScore <- read.csv("C:/Users/Wen/Desktop/STAT 628/module2/sentimentScore.csv")
testset <- read.csv("C:/Users/Wen/Desktop/STAT 628/module2/testval_data.csv")

categories <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/categories_dict.csv", col_names = FALSE)
categories <- categories[, -3]  #without restaurants

SUB <- train_data[1:20000,]

INDEX <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/INDEX.csv")

## Food indexing
#MILK
set = c()
k = 1 
for (i in 1:20000){
  if(str_detect( strsplit(SUB$text[i]," "),"milk") == TRUE ){
    set[k] = i
    k = k+1
  }
  
}
milk = SUB[set ,]

Toplot = milk%>%
  group_by(stars) %>%
  summarise(n = n())
Toplot = as.data.frame(Toplot)



#cREAM
set = c()
k = 1 
for (i in 1:20000){
  if(str_detect( strsplit(SUB$text[i]," "),"cream") == TRUE ){
    set[k] = i
    k = k+1
  }
  
}
cream = SUB[set ,]

Toplot2 = cream%>%
  group_by(stars) %>%
  summarise(n = n())
Toplot2 = as.data.frame(Toplot2)

#Cheese
set = c()
k = 1 
for (i in 1:20000){
  if(str_detect( strsplit(SUB$text[i]," "),"cheese") == TRUE ){
    set[k] = i
    k = k+1
  }
  
}
cheese = SUB[set ,]

Toplot3 = cheese%>%
  group_by(stars) %>%
  summarise(n = n())
Toplot3 = as.data.frame(Toplot3)

#eggs
set = c()
k = 1 
for (i in 1:20000){
  if(str_detect( strsplit(SUB$text[i]," "),"eggs") == TRUE ){
    set[k] = i
    k = k+1
  }
  
}
eggs = SUB[set ,]

Toplot4 = eggs%>%
  group_by(stars) %>%
  summarise(n = n())
Toplot4 = as.data.frame(Toplot4)



#install.packages(c("ggplot2","Rmisc","lattice","plyr"))

library(ggplot2)
library(Rmisc)
library(lattice)
library(plyr)


p1 = ggplot(Toplot, aes(x=stars, y=n)) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "milk")

p2 = ggplot(Toplot2, aes(x=stars, y=n)) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "cream")

p3 = ggplot(Toplot3, aes(x=stars, y=n)) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "cheese")

p4 = ggplot(Toplot4, aes(x=stars, y=n)) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "eggs")

multiplot(p1,p2,p3,p4,cols = 2)



# Exclamation
exclamation <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/plot/num_exclamation_mark.csv")
one <- exclamation%>%filter(stars == 1)%>% 
            filter(num_exclamation_mark != 0) %>% 
              filter(num_exclamation_mark != 1) 
two <- exclamation%>%filter(stars == 2)%>% 
  filter(num_exclamation_mark != 0) %>% 
  filter(num_exclamation_mark != 1) 
three <- exclamation%>%filter(stars == 3)%>% 
  filter(num_exclamation_mark != 0) %>% 
  filter(num_exclamation_mark != 1) 
four <- exclamation%>%filter(stars == 4)%>% 
  filter(num_exclamation_mark != 0) %>% 
  filter(num_exclamation_mark != 1) 
five <- exclamation%>%filter(stars == 5)%>% 
  filter(num_exclamation_mark != 0) %>% 
  filter(num_exclamation_mark != 1) 

mean = exclamation %>% group_by(stars) %>%
                      summarise(mean = sum(multiply)/sum(count))
                       

q1 = ggplot(one, aes(x=num_exclamation_mark, y=count/sum(count))) + 
  geom_bar(stat="identity", fill="lightblue", colour="black") + labs(title = "one star")+xlim(0,125)+ylim(0,0.5)+ylab("Count")

q2 = ggplot(two, aes(x=num_exclamation_mark, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "two star")+xlim(0,125)+ylim(0,0.5)+ylab("Count")
q3 = ggplot(three, aes(x=num_exclamation_mark, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "three star")+xlim(0,125)+ylim(0,0.5)+ylab("Count")
q4 = ggplot(four, aes(x=num_exclamation_mark, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "four star")+xlim(0,125)+ylim(0,0.5)+ylab("Count")
q5 = ggplot(five, aes(x=num_exclamation_mark, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "five star")+xlim(0,125)+ylim(0,0.5)+ylab("Count")


multiplot(q1,q2,q3,q4,q5,cols = 1)





# Upperwords

upperwords <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/plot/num_upper_words.csv")
one <- upperwords%>%filter(stars == 1)%>% 
  filter(num_upper_words != 0) 
 
two <- upperwords%>%filter(stars == 2)%>% 
  filter(num_upper_words != 0) 
 
three <- upperwords%>%filter(stars == 3)%>% 
  filter(num_upper_words != 0) 
 
four <- upperwords%>%filter(stars == 4)%>% 
  filter(num_upper_words != 0) 
  
five <- upperwords%>%filter(stars == 5)%>% 
  filter(num_upper_words != 0) 
  


q1 = ggplot(one, aes(x=num_upper_words, y=count/sum(count))) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "one star")+xlim(0,125)+ylim(0,0.5)

q2 = ggplot(two, aes(x=num_upper_words, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "two star")+xlim(0,125)+ylim(0,0.5)
q3 = ggplot(three, aes(x=num_upper_words, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "three star")+xlim(0,125)+ylim(0,0.5)
q4 = ggplot(four, aes(x=num_upper_words, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "four star")+xlim(0,125)+ylim(0,0.5)
q5 = ggplot(five, aes(x=num_upper_words, y=count/sum(count))) +
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "five star")+xlim(0,125)+ylim(0,0.5)


multiplot(q1,q2,q3,q4,q5,cols = 2)



##  certain food trend 

# Special distrbution

foodcount <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/plot/foodcount.csv")

index <- c(1,2,3,4,5)

#normal distribution



#drink
cocoa <- data.frame(name = index, count = foodcount$cocoa[9:13])
spices <- data.frame(name = index, count = foodcount$spices[9:13])
water <-data.frame(name = index, count = foodcount$water[9:13])
spirits <-data.frame(name = index, count = foodcount$spirits[9:13])

#fruit
melons <-data.frame(name = index, count = foodcount$melons[9:13])
lemons <-data.frame(name = index, count = foodcount$lemons[9:13])

#oil
margarine <- data.frame(name = index, count = foodcount$margarine[9:13])

#vegetables
cauliflower <- data.frame(name = index, count = foodcount$cauliflower[9:13])
celery <- data.frame(name = index, count = foodcount$celery[9:13])
escarole <- data.frame(name = index, count = foodcount$escarole[9:13])

drink1 = ggplot(cocoa, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "cocoa") + theme(axis.text.y = element_blank()) 
#drink2 = ggplot(spices, aes(x=index, y=count)) + 
#  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "specis") + theme(axis.text.y = element_blank()) 
drink3 = ggplot(water, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "water") + theme(axis.text.y = element_blank()) 
drink4 = ggplot(spirits, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "spirits") + theme(axis.text.y = element_blank()) 

multiplot(drink1,drink3,drink4,cols = 2)

#fruit
fruit1 =  ggplot(melons, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "melons") + theme(axis.text.y = element_blank()) 
fruit2 =  ggplot(lemons, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "lemons") + theme(axis.text.y = element_blank()) 

multiplot(fruit1,fruit2,cols = 1)

#oil
ggplot(margarine, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "margarine") + theme(axis.text.y = element_blank()) 

#vegetaables
veg1 = ggplot(cauliflower, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "cauliflower") + theme(axis.text.y = element_blank()) 
veg2 = ggplot(celery, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "celery") + theme(axis.text.y = element_blank()) 
veg3 = ggplot(escarole, aes(x=index, y=count)) + 
  geom_bar(stat="identity", fill="lightblue", colour="black")+ labs(title = "escarole") + theme(axis.text.y = element_blank()) 

multiplot(veg1,veg2,veg3,cols = 2)




# map
#install.packages("zipcode")
library(choroplethr)
library(choroplethrMaps)
library(tidyverse)
library(zipcode)
library(maps)
#1

onestar = train_data%>%filter(stars == 1)
data("zipcode")
onestar2 = onestar %>% mutate(region = zipcode$zip[match(city,zipcode$city)])
onestar2 <- onestar2 %>%
  filter(!is.na(region))

onestar2$region = as.integer(onestar2$region)
points <- onestar2 %>% 
  group_by(region) %>%
  dplyr::summarize(value = n())
  
choro = CountyChoropleth$new(points)
choro$title = "One-star restaurants Distribution"
choro$ggplot_scale = scale_fill_brewer(name="Number of Restaurants", drop=FALSE)
choro$render()


#data(df_pop_county)
#choro <- county_choropleth(points)


#2 stars
twostar = train_data%>%filter(stars == 2)
data("zipcode")
twostar2 = twostar %>% mutate(region = zipcode$zip[match(city,zipcode$city)])
twostar2 <- twostar2 %>%
  filter(!is.na(region))

twostar2$region = as.integer(twostar2$region)
points <- twostar2 %>% 
  group_by(region) %>%
  dplyr::summarize(value = n())

choro = CountyChoropleth$new(points)
choro$title = "Two-star restaurants Distribution"
choro$ggplot_scale = scale_fill_brewer(name="Number of Restaurants", drop=FALSE)
choro$render()

length(unique(train_data$city))


#all 
total <- train_data
data("zipcode")
data("us.cities")
total = total %>% mutate(region = zipcode$zip[match(city,zipcode$city)])
total <- total%>%
  filter(!is.na(region))

#state_choropleth(points)

points <- total %>% 
  group_by(region) %>%
  dplyr::summarize(value = n())


points$region = as.integer(points$region)
#choroplethr(points, lod="state")

choro = CountyChoropleth$new(points)
choro$title = " Restaurants Distribution"
choro$ggplot_scale = scale_fill_brewer(name="Number of Comments", drop=FALSE)
choro$render()


length(unique(train_data$city))

install.packages("maps")
library(maps)

change = data.frame(fullname = continental_us_states,simple = full)
data(continental_us_states)

################
state <- read_csv("C:/Users/Wen/Desktop/STAT 628/module2/state.csv", 
                  col_names = FALSE)
data("zipcode")
total <- train_data  
total = total %>% mutate(region = zipcode$state[match(city,zipcode$city)])
total <- total%>%
  filter(!is.na(region))
points <- total %>% 
  group_by(region) %>%
  dplyr::summarize(value = n())

points = points %>% mutate(region2 = state$X3[match(region,state$X2)])

newpoints = data.frame(region = c(as.vector(points$region2),c(rep("BB",16))),value = c( points$value,c(rep(0,16)) ) )
newpoints$region = as.vector(newpoints$region)

k=1

for (i in 1:51){
  
  if (!(df_pop_state$region[i] %in% newpoints$region)){
    newpoints$region[36+k] = df_pop_state$region[i]
    k=k+1
  }
}


state_choropleth(newpoints, 
                 title  = "Comments Distribution", 
                 legend = "Count")
