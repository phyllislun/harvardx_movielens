####################################
# HarvardX PH125.9x Data Science: Capstone
# MovieLens Project Submission
# Author: Phyllis Lun
# Date date: 20220308
#
###################################

##########################################################
# Create edx set, validation set (final hold-out test set)---------------

##########################################################


# Note: this process could take a couple of minutes

#You will use the following code to generate your datasets. Develop your algorithm using the edx set. 
#For a final test of your final algorithm, predict movie ratings in the validation set (the final hold-out test set) 
#as if they were unknown. RMSE will be used to evaluate how close your predictions are to the true values in the 
#validation set (the final hold-out test set).

#Important: The validation data (the final hold-out test set) should NOT be used for training, developing, 
#or selecting your algorithm and it should ONLY be used for evaluating the RMSE of your final algorithm. 
#The final hold-out test set should only be used at the end of your project with your final model. 
#It may not be used to test the RMSE of multiple models during model development. 
#You should split the edx data into separate training and test sets to design and test your algorithm.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(knitr)
library(stats)
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
getOption('timeout')
options(timeout=600)
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)


ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set-- so there are ioveralaps in both sets
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

dim(validation)
dim(edx)
head(edx, n=3)

######

library(tidyverse)
library(caret)
library(data.table)
library(stringr)
library(stats)
library(scales)
#library(lucidate)



#Data cleaning-----------------
#Training set------------------
#Here I first separated the movie titles from the year of movie release. I have also converted the time stamps of review to date format and extracted the years or rating. 
#A new variable, year_diff, was used to assess the time difference between year of movie release and year of rating. 

edx_clean <- edx%>% 
  mutate(year=str_extract(title, "\\(\\d+\\)")) %>% mutate(year=str_extract(year, "\\d+")) %>% 
  mutate(title=str_remove(title, "\\s+\\(\\d+\\)")) %>% #cleaning movie title and year 
  mutate(rate_year= as.POSIXct(timestamp,origin = "1970-01-01"))%>% #converting timestamps to actual date, and extract the year of rating and the year the movie was released 
  mutate(rate_year=as.numeric(format(rate_year,"%Y")), year=as.numeric(str_extract(year,"\\d+"))) %>% 
  select(-timestamp) %>% mutate(year_diff=rate_year-year)

#tackling the one and only movie with two brackets of years in the title ---
edx_clean[edx_clean$movieId==889]<-edx_clean[edx_clean$movieId==889] %>% mutate (year=as.numeric(1994))   %>% mutate(year_diff=rate_year-year)



# A similar data processing procedure was done to the final validation data. 
validation_clean<-validation%>% 
  mutate(year=str_extract(title, "\\(\\d+\\)")) %>% mutate(year=str_extract(year, "\\d+")) %>% 
  mutate(title=str_remove(title, "\\s+\\(\\d+\\)")) %>% #cleaning movie title and year 
  mutate(rate_year= as.POSIXct(timestamp,origin = "1970-01-01"))%>% #converting timestamps to actual date, and extract the year of rating and the year the movie was released 
  mutate(rate_year=as.numeric(format(rate_year,"%Y")), year=as.numeric(str_extract(year,"\\d+"))) %>% 
  select(-timestamp) %>% mutate(year_diff=rate_year-year)

validation_clean[validation$movieId==889]<-validation_clean[validation_clean$movieId==889] %>% 
  mutate (year=as.numeric(1994))  %>% mutate(year_diff=rate_year-year)


##Data exploration: edx data set-----------------
edx_clean %>% summarize(n_users = n_distinct(userId), n_movies = n_distinct(movieId))
#In the edx dataset, we have 69878 unique users and 10677 unique movies.

#n_users 69878
#n_movies 10677

#1. Distribution of movie ratings---- 
edx_clean %>% ggplot (aes(x=rating)) + 
  geom_bar()+  
  labs(title="Distributions of movie ratings",x="Ratings", y="Count", face="bold")+
  scale_x_continuous()+
  scale_y_discrete(labels = scales::comma)+
  theme_classic()
#The most frequently given ratings are 3 and 4. 

#2. Distribution of number of ratings by users---
edx_clean %>% group_by(userId) %>% count(userId)  %>% summary(n) 
#We can see that there is a huge variation in the number of movie ratings given by the users (range=10-6616). 
#Three quarters of the users rated 141 movies or less. 

#Of the 69878 users included in the pilot set, over average they have 129 move ratings and three quarters of them have about 141 ratings. 
#That means a quarter of users gave a large number of movie ratings. 


#Figure 2.1-- not shown in report 
edx_clean %>% group_by(userId) %>% count(userId) %>% ggplot(aes(x=n)) + geom_bar() +
  labs(title="Number of movie rating by users (full sample)",x="Ratings", y="Count", face="bold")+
  theme_classic()

#Figure 2.2
edx_clean %>% group_by(userId) %>% count(userId)  %>% filter(n<=141)  %>% ggplot(aes(x=n)) + geom_bar() + 
  labs(title="Number of movie rating by users (3 quarters of the sample)",x="Ratings", y="Count", face="bold")+
  theme_classic()


# Figure 2.3 Association between the variations of movie ratings and prolific raters
edx_clean %>% group_by(userId) %>% summarize(standard_dev=sd(rating)) %>% left_join(edx_clean %>% group_by(userId) %>% count(userId))%>% 
  filter(n<=141) %>% as_tibble() %>% 
  ggplot(aes(x=n,y=standard_dev)) + geom_point(size=0.1) + 
  scale_x_continuous()+
  labs(title="Standard deviations of movie ratings by number of user ratings",x="Ratings", 
       y="standard deviations", face="bold")+
  theme_classic()
#Generally, ratings coming from prolific raters are more similar than those from users who provided only a few ratings.However, it should be noted that the deviation is quite large.  

#3. Number of ratings received by a movie 
edx_clean %>% dplyr::count(movieId) %>% summary()
#Similar to the number of ratings by users, the number of ratings received by movies also varied greatly (range = 1-31362). 
# Only three quarters of movies received less than 565 ratings. 

#Figure 3.1-- not shown in report 
edx_clean %>% dplyr::count(movieId) %>% ggplot(aes(x=n)) + geom_bar() +
  labs(title="Number of ratings received by a movie (full sample)",x="Ratings", y="Count", face="bold")+
  theme_classic()

#Figure 3.2-- not shown in report 
edx_clean %>% dplyr::count(movieId) %>% filter(n<=565.0) %>% ggplot(aes(x=n)) + geom_bar() +
  labs(title="Number of ratings received by a movie (3 quarters of the sample)",x="Ratings", y="Count", face="bold")+
  theme_classic()

#Figure 3.3 Year of movie release and year of movie rating 
#Correlation between rating and years since movie release

#It is noted that the correlation between the year of rating and the year of movie release appears roughly normally distributed
edx_clean %>% group_by(movieId) %>% mutate(rating=as.numeric(rating)) %>% 
  summarize(correlations=cor(year_diff,rating)) %>%
  ggplot (aes(correlations)) + 
  geom_histogram(bins =250)+ 
  labs(x = "Correlation coefficients of year of\nmovie release and year of rating")+
  #scale_y_discrete(name="Count")+ 
  theme_bw()

#Model training---------
#Evaluation metric: RMSE-----
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#### Within the edx dataset, creating a training set and a test set 
set.seed(125, sample.kind = "Rounding")
edx_test_index <- createDataPartition(edx_clean$rating, times = 1, p = 0.1, list = FALSE)
edx_clean_train <- edx_clean[-edx_test_index,]
edx_clean_test<- edx_clean[edx_test_index,]
#pilot_x<-pilot %>% select(-rating)
#pilot_y<-pilot$rating
nrow(edx_clean_train) #8100048


#Model 1: Just movie ID 
mu <- mean(edx_clean_train$rating)
avgs_movie <- edx_clean_train %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- edx_clean_test  %>% 
  left_join(avgs_movie, by='movieId') %>%
  mutate(prediction = mu + b_i) 

sum(is.na(predicted_ratings$prediction))
predicted_ratings$prediction[is.na(predicted_ratings$prediction)]<-0
model_1_rmse <- RMSE(edx_clean_test$rating, predicted_ratings$prediction)

edx_rmse_results <- tibble(method = "edx: movidId only model", RMSE = model_1_rmse)
edx_rmse_results %>% knitr::kable()


#Model 2: Just movie ID and user ID (i.e., adding user ID)
avgs_user <- edx_clean_train %>% 
  left_join(avgs_movie, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- edx_clean_test %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  mutate(prediction = mu + b_i + b_u) 

sum(is.na(predicted_ratings$prediction))
predicted_ratings$prediction[is.na(predicted_ratings$prediction)]<-0

model_2_rmse <- RMSE(edx_clean_test$rating, predicted_ratings$prediction)
edx_rmse_results <-bind_rows(edx_rmse_results, data.frame(method="edx: movidId and userId model", RMSE=model_2_rmse))
edx_rmse_results %>% knitr::kable()

#Model 3: adding difference in year of rating and year of movie release
avgs_time <- edx_clean_train %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  group_by(year_diff) %>%
  summarize(b_v = mean(rating - mu-b_i-b_u))

predicted_ratings <- edx_clean_test %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_time, by='year_diff') %>%
  mutate(prediction = mu + b_i + b_u+b_v) 


sum(is.na(predicted_ratings$prediction))
predicted_ratings$prediction[is.na(predicted_ratings$prediction)]<-0

model_3_rmse <- RMSE(edx_clean_test$rating, predicted_ratings$prediction)
edx_rmse_results <-bind_rows(edx_rmse_results, data.frame(method="edx: movidId, userId,and difference in year model", RMSE=model_3_rmse))
edx_rmse_results %>% knitr::kable()


#Model 4: adding all the genres
avgs_genres <- edx_clean_train %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_time, by='year_diff') %>%
  group_by(genres) %>%
  summarize(b_w = mean(rating - mu-b_i-b_u-b_v))

predicted_ratings <- edx_clean_test %>% 
  left_join(avgs_movie, by='movieId') %>%
  left_join(avgs_user, by='userId') %>%
  left_join(avgs_time, by='year_diff') %>%
  left_join(avgs_genres, by='genres') %>%
  mutate(prediction = mu + b_i + b_u+b_v+b_w) 

sum(is.na(predicted_ratings$prediction))
predicted_ratings$prediction[is.na(predicted_ratings$prediction)]<-0

model_4_rmse <- RMSE(edx_clean_test$rating, predicted_ratings$prediction)
edx_rmse_results <-bind_rows(edx_rmse_results, data.frame(method="edx: movidId, userId, difference in year, and genre model", RMSE=model_4_rmse))
edx_rmse_results %>% knitr::kable()



# Regularization: Tuning for the lambda----
lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
  mu <- mean(edx_clean_train$rating)
  
  
  b_i <- edx_clean_train %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
 
   b_u <- edx_clean_train %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
   b_v <- edx_clean_train %>% 
     left_join(b_i, by='movieId') %>%
     left_join(b_u, by='userId') %>%
     group_by(year_diff) %>%
     summarize(b_v = mean(rating - mu-b_i-b_u)/(n()+l))
   
   b_w <- edx_clean_train %>% 
     left_join(b_i, by='movieId') %>%
     left_join(b_u, by='userId') %>%
     left_join(b_v, by='year_diff') %>%
     group_by(genres) %>%
     summarize(b_w = mean(rating - mu-b_i-b_u-b_v)/(n()+l))
  
  
  predicted_ratings <- edx_clean_test %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_v, by='year_diff') %>%
    left_join(b_w, by='genres') %>%
    mutate(prediction = mu + b_i + b_u+b_v+b_w) 
  
  predicted_ratings$prediction[is.na(predicted_ratings$prediction)]<-0
  
  return(RMSE(edx_clean_test$rating, predicted_ratings$prediction))
})


qqplot(lambdas, rmses)  
lambdas[which.min(rmses)]
#It turns out of lamda regularization equal 5 is needed to achieve the lowest RMSE.


##Applying the regularized regression model 4 to validation set----- 

#Regularized Model 4---
mu <- mean(edx_clean_train$rating)
l<-5

b_i <- edx_clean_train %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+l))

b_u <- edx_clean_train %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+l))

b_v <- edx_clean_train %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(year_diff) %>%
  summarize(b_v = mean(rating - mu-b_i-b_u)/(n()+l))

b_w <- edx_clean_train %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_v, by='year_diff') %>%
  group_by(genres) %>%
  summarize(b_w = mean(rating - mu-b_i-b_u-b_v)/(n()+l))


predicted_ratings <- validation_clean %>% 
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_v, by='year_diff') %>%
  left_join(b_w, by='genres') %>%
  mutate(prediction = mu + b_i + b_u+b_v+b_w) 


sum(is.na(predicted_ratings$prediction))
predicted_ratings$prediction[is.na(predicted_ratings$prediction)]<-0

model_4_rmse <- RMSE(validation_clean$rating, predicted_ratings$prediction)
edx_rmse_results <-bind_rows(edx_rmse_results, data.frame(method="validation: movidId, userId, difference in year, and genre model (regularized)", RMSE=model_4_rmse))
edx_rmse_results %>% knitr::kable()


#After fitting the trained linear regression models to the validation data, the model with the best performance is the model with movidId and userId only as features, 
#which yields a RMSE of 0.86522840.