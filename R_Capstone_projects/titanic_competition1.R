################################# Predicting Survival in the Titanic ##################################
# Load necessary libraries
library(randomForest)
library(pROC)
library(cvTools)
library(dplyr)

#set directory
setwd("~/Documents/competition_data")
train_titanic <- read.csv("train.csv", stringsAsFactors = F)
test_titanic <- read.csv("test.csv", stringsAsFactors = F)

test_titanic$Survived <- NA
test_titanic$data <- "test"
train_titanic$data <- "train"


# combine train and train and test data set for cleaning and
#preprocessing process

titan_data <- rbind(train_titanic, test_titanic)

#library(VIM)
#VIM::aggr(titan_data,numbers = T, prop = c(T,F))
#aggr(biopics, numbers = TRUE, prop = c(TRUE, FALSE))

#lapply(titan_data, function(x) sum(is.na(x)))

#length(titan_data$Age[titan_data$Age == "NA"])# 263 NAs

table(titan_data$Sex)
View(titan_data)

titan_data1 <- titan_data

#treatment of missing values
titan_data1[is.na(titan_data1[,"Age"]), "Age"] <- median(titan_data1[,"Age"], na.rm = T)

lapply(titan_data1, function(x) sum(is.na(x)))

titan_data1[is.na(titan_data1[,"Fare"]), "Fare"] <- mean(titan_data1[,"Fare"],na.rm = T)

lapply(titan_data1, function(x) sum(is.na(x)))

titan_data2 <- titan_data1 %>% select(-Name,-Ticket,-Cabin)

prop.table(table(titan_data2$Embarked))

round(tapply(titan_data2$Survived, titan_data2$Embarked,function(x) mean(x,na.rm = T)),2)


Mode <- function(x){
  a = x[!is.na(x)]
  ux = unique(a)
  mode = ux[which.max(tabulate(match(x,ux)))]
  return(mode)
}

titan_data3 <- titan_data2



titan_data3$Embarked[titan_data3$Embarked == ""] <- Mode(titan_data3[,"Embarked"])
str(titan_data3)

titan_data4 <- titan_data3 %>% mutate(Sex = ifelse(Sex == "female", 1, 0),
                                      Sex = as.numeric(Sex))
str(titan_data4)

#titan_data4 <- titan_data3 <- mutate(Sex = ifelse(Sex ==  "male", 1,0),
                                     #Sex = as.numeric(Sex))


CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}


 for ( var in c("Embarked")){
   titan_data5 = CreateDummies(titan_data4,var, 100)
 }

glimpse(titan_data5)
library(dplyr)


glimpse(titan_data5)

lapply(titan_data5, function(x) sum(is.na(x)))

#seperate data back to test and train to build model and do prediction accordingly

train_data_titan <- titan_data5 %>% filter(data == "train") %>% select(-data)
test_data_titan <- titan_data5 %>% filter(data == "test") %>% select(-data, -Survived)

lapply(test_data_titan, function(x) sum(is.na(x)))
View(train_data_titan)
# check and remove multicolinearity
install.packages("caret")
install.packages("car")
library(car)
library(caret)

prop.table(table(train_data_titan$Survived))

for_vif <- lm(Survived~.-PassengerId, data = train_data_titan)
sort(vif(for_vif), decreasing = T)

train_data_titan$Survived <- as.factor(train_data_titan$Survived)

random_f_fit <- glm(Survived~.-PassengerId -Embarked_S -Embarked_C -Parch -Fare, data = train_data_titan, family = "binomial")

summary(random_f_fit)

pred_train <- predict(random_f_fit, newdata = train_data_titan, type = "response")

pROC::roc(train_data_titan$Survived, pred_train) #.85

# treat data imbalance with upsampling












up_samp_data <- upSample(x = train_data_titan[,colnames(train_data_titan) != "Survived"],
                       y = as.factor(train_data_titan$Survived))

table(up_samp_data$Class)

random_f_fit_up <- glm(Class~.-PassengerId -Embarked_C -Embarked_S -Fare -Parch, data = up_samp_data, family = "binomial")

pred_up <- predict(random_f_fit_up, newdata = up_samp_data, type = "response")

pROC::roc(up_samp_data$Class, pred_up) #.85

train_data_titan$Survived <- as.factor(train_data_titan$Survived)


forest_fit <- randomForest(Survived~.-PassengerId -Embarked_S -Embarked_C -Parch -Fare, data = train_data_titan, mtry = 35, ntree = 900, maxnodes = 20, nodesize = 10)

ran_pred <- predict(forest_fit, newdata = train_data_titan, type = "prob")[,2]

pROC::roc(train_data_titan$Survived, ran_pred) #.85, 0.91

# crossvalidation and parameter tuning
varImpPlot(forest_fit)

param <- list(mtry = c(5,10,15,20,25,35,40,50),
              ntree = c(200,500,700,900,1000),
              maxnodes = c(15,20,30,50,100,150,200),
              nodesize = c(1,2,5,10,15,20))

mycost_auc <- function(y, yhat){
  roccurve = pROC::roc(y, yhat)
  score = pROC::auc(roccurve)
  return(score)
}

num_trials <- 336

subset_para <- function(full_list_para,n=10){
  all_comb = expand.grid(full_list_para)
  s = sample(1:nrow(all_comb),n)
  subset_para = all_comb[s,]
  return(subset_para)
}

my_params <- subset_para(param,num_trials)



myauc = 0


for (i in 1:num_trials){
  print(i)
  params_to_be_used = my_params[i,]
  ran_ref_fit <- cvTuning(randomForest,
                          Survived ~.-PassengerId -Embarked_C -Embarked_S -Fare -Parch,
                          data = train_data_titan,
                          tuning = params_to_be_used,
                          folds = cvFolds(nrow(train_data_titan),K=10,type = c("random")),
                          cost = mycost_auc, seed = 2,
                          predictArgs = list(type = c("prob")))
  current_auc = ran_ref_fit$cv[,2]
  if(current_auc > myauc){
    myauc = current_auc
    print(params_to_be_used)
    best_params = params_to_be_used
  }
}

#get cutoff

?cvTuning
real <- train_data_titan$Survived
#0.001, 0.999, 0.001

cutoffs <- seq(0.001,0.999,0.001)

cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(ran_pred > cutoff)
  
  TP =sum(real == 1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real == 1 & predicted == 0)
  
  
  
  P = TP + FN
  N = TN + FP
  
  Sn = TP/P
  Sp = TN/N
  precision = TP/(TP + FP)
  recall = Sn
  KS = (TP/P) - (FP/N)
  F5 =(26*precision*recall)/((25*precision) + recall)
  F.1 =(1.01*precision*recall)/((.01*precision) + recall)
  M = (4*FP+FN)/(5*(P+N))
  cutoff_data = rbind(cutoff_data,
                      c(cutoff,Sn,Sp,KS,F5,F.1,M))
  
}

cutoff_data = cutoff_data[-1,]

library(ggplot2)
prop.table(table(shad_train_titan$Survived))

library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

ggplot(cutoff_data, aes(x =cutoff, y = KS)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff

test_pred <- predict(forest_fit, newdata = test_data_titan, type = "prob")[,2]

hard_class_test <- as.numeric(test_pred > 0.25)
?write.csv
write.csv(hard_class_test,"predict_survived.csv", row.names = T )
################################################### END   ############################################

######################################### use boosting and adaboost#####################################
install.packages("adabag")
library(adabag)
?boosting

#check or iterate for the best parameter

#boosting model

boo_fit <- boosting(Survived~.-PassengerId -Embarked_S -Embarked_C -Parch -Fare, data = train_data_titan, boos = TRUE, mfinal = 300)

test_pred <- predict(boo_fit, newdata = test_data_titan)
train_pred <- predict(boo_fit, newdata = train_data_titan)
prob_train_pred <- train_pred$prob[,2]
summary(test_pred)

library(pROC)
pROC::roc(train_data_titan$Survived, prob_train_pred)

#check for cutoff
real <- train_data_titan$Survived
#0.001, 0.999, 0.001

cutoffs <- seq(0.1,0.9,0.05)

cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(prob_train_pred>cutoff)
  
  TP =sum(real == 1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real == 1 & predicted == 0)
  
  
  
  P = TP + FN
  N = TN + FP
  
  Sn = TP/P
  Sp = TN/N
  precision = TP/(TP + FP)
  recall = Sn
  KS = (TP/P) - (FP/N)
  F5 =(26*precision*recall)/((25*precision) + recall)
  F.1 =(1.01*precision*recall)/((.01*precision) + recall)
  M = (4*FP+FN)/(5*(P+N))
  cutoff_data = rbind(cutoff_data,
                      c(cutoff,Sn,Sp,KS,F5,F.1,M))
  
}

cutoff_data = cutoff_data[-1,]

library(ggplot2)
prop.table(table(shad_train_titan$Survived))

library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

ggplot(cutoff_data, aes(x =cutoff, y = KS)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff

prob_pred <- test_pred$prob[,2]
cutof_class_test <-  as.numeric(prob_pred > 0.45)

write.csv(cutof_class_test,"adatitanic.csv", row.names = TRUE)


################################## END OF ADBOOST MODEL ####################################
# build xgboost

install.packages("gbm")

install.packages("xgboost")
library(xgboost)
xtrain <- train_data_titan %>% select(-Survived)
ytrain <- train_data_titan$Survived
ytrain <- as.numeric(as.factor(ytrain)) - 1
#let get the best params for the model
?xgboost
#sey <-seq(.01,.3,.01)
param <- list(max_depth = c(5,6,10,15,20,25,35),
              gamma = c(50,100,200,500,700, 900,1000),
              nrounds = c(5,10,15,20,30,50,100),
              eta = c(0.01,0.06,0.10,0.15,0.17,0.25,0.30),
              min_child_weight = c(1,2,5,6,7,10),
              subsample = c(0.5,0.6,0.7,0.8),
              cosample_bytree = c(0.5,0.6,0.7,0.8,0.9),
              max_delta_step = c(1,2,5,10))

mycost_auc <- function(y, yhat){
  roccurve = pROC::roc(y, yhat)
  score = pROC::auc(roccurve)
  return(score)
}

num_trials <- 500

subset_para <- function(full_list_para,n=10){
  all_comb = expand.grid(full_list_para)
  s = sample(1:nrow(all_comb),n)
  subset_para = all_comb[s,]
  return(subset_para)
}

my_params <- subset_para(param,num_trials)



myauc = 0


for (i in 1:num_trials){
  print(i)
  params_to_be_used = my_params[i,]
  ran_ref_fit <- cvTuning(xgboost,
                          data = data.matrix(xtrain),
                          label = ytrain,
                          objective = "reg:logistic",
                          tuning = params_to_be_used,
                          folds = cvFolds(nrow(xtrain),K=10,type = c("random")),
                          cost = mycost_auc, seed = 2,
                          predictArgs = list(type = c("prob")))
  current_auc = ran_ref_fit$cv[,2]
  if(current_auc > myauc){
    myauc = current_auc
    print(params_to_be_used)
    best_params = params_to_be_used
  }
}



extrem_b_model <- xgboost(data = data.matrix(xtrain),
                          label = ytrain,
                          objective = "reg:logistic",
                          nrounds = 100)

train_pred <- predict(extrem_b_model,data.matrix(xtrain))
test_pred <- predict(extrem_b_model, data.matrix(test_data_titan))

pROC::roc(ytrain, train_pred)

#get cutoff

real <- ytrain
#0.001, 0.999, 0.001

cutoffs <- seq(0.1,0.9,0.05)

cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(train_pred>cutoff)
  
  TP =sum(real == 1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real == 1 & predicted == 0)
  
  
  
  P = TP + FN
  N = TN + FP
  
  Sn = TP/P
  Sp = TN/N
  precision = TP/(TP + FP)
  recall = Sn
  KS = (TP/P) - (FP/N)
  F5 =(26*precision*recall)/((25*precision) + recall)
  F.1 =(1.01*precision*recall)/((.01*precision) + recall)
  M = (4*FP+FN)/(5*(P+N))
  cutoff_data = rbind(cutoff_data,
                      c(cutoff,Sn,Sp,KS,F5,F.1,M))
  
}

cutoff_data = cutoff_data[-1,]

library(ggplot2)
prop.table(table(shad_train_titan$Survived))

library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

ggplot(cutoff_data, aes(x =cutoff, y = KS)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff



test_pred_outcome <- as.numeric(test_pred > 0.4)

write.csv(test_pred_outcome, "xgtitanic.csv", row.names = T)



shadow_data <- train_data_titan





#for_vif <- lm(Survived ~.-PassengerId, data = train_data_titan)

#partition the training data into two so as to have a test training se to evaluate prediction on
# unseen data


shad_data_index <- createDataPartition(shadow_data$Survived, p = 0.90, list = FALSE)
shad_train_titan <- shadow_data[shad_data_index,]
shad_test_titan <- shadow_data[-shad_data_index,]


# create oversampling data set


# check for class imbalance

table(shad_train_titan$Survived)


u_sample_d <- upSample(x = shad_train_titan[,colnames(shad_train_titan) != "Survived"],
                       y = as.factor(shad_train_titan$Survived))
table(u_sample_d$Class)

random_model_up <- randomForest(Class ~.-PassengerId -Embarked_C -Embarked_S -Fare -Parch,
                             data = u_sample_d,mtry = 5, ntree = 500, maxnodes = 100, nodesize = 1)

pred_up_sam <- predict(random_model_up, newdata = u_sample_d, type = "prob")[,2]

auc(roc(u_sample_d$Class, pred_up_sam))

pred_up_sam_t <- predict(random_model_up, newdata = shad_test_titan, type = "prob")[,2]
 pROC::roc(shad_test_titan$Survived, pred_up_sam_t)

#tune upsamble data

param <- list(mtry = c(5,10,15,20,25,35),
              ntree = c(50,100,200,500,700),
              maxnodes = c(5,10,15,20,30,50,100),
              nodesize = c(1,2,5,10))

mycost_auc <- function(y, yhat){
  roccurve = pROC::roc(y, yhat)
  score = pROC::auc(roccurve)
  return(score)
}

num_trials <- 120

subset_para <- function(full_list_para,n=10){
  all_comb = expand.grid(full_list_para)
  s = sample(1:nrow(all_comb),n)
  subset_para = all_comb[s,]
  return(subset_para)
}

my_params <- subset_para(param,num_trials)



myauc = 0


for (i in 1:num_trials){
  print(i)
  params_to_be_used = my_params[i,]
  ran_ref_fit <- cvTuning(randomForest,
                          Class ~.-PassengerId -Embarked_C -Embarked_S -Fare -Parch,
                          data = u_sample_d,
                          tuning = params_to_be_used,
                          folds = cvFolds(nrow(u_sample_d),K=10,type = c("random")),
                          cost = mycost_auc, seed = 2,
                          predictArgs = list(type = c("prob")))
  current_auc = ran_ref_fit$cv[,2]
  if(current_auc > myauc){
    myauc = current_auc
    print(params_to_be_used)
    best_params = params_to_be_used
  }
}

# get cut off for upsample data

real <- u_sample_d$Class
#0.001, 0.999, 0.001

cutoffs <- seq(0.1,0.9,0.05)

cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(pred_up_sam>cutoff)
  
  TP =sum(real == 1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real == 1 & predicted == 0)
  
  
  
  P = TP + FN
  N = TN + FP
  
  Sn = TP/P
  Sp = TN/N
  precision = TP/(TP + FP)
  recall = Sn
  KS = (TP/P) - (FP/N)
  F5 =(26*precision*recall)/((25*precision) + recall)
  F.1 =(1.01*precision*recall)/((.01*precision) + recall)
  M = (4*FP+FN)/(5*(P+N))
  cutoff_data = rbind(cutoff_data,
                      c(cutoff,Sn,Sp,KS,F5,F.1,M))
  
}

cutoff_data = cutoff_data[-1,]

library(ggplot2)
prop.table(table(shad_train_titan$Survived))

library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

ggplot(cutoff_data, aes(x =cutoff, y = KS)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff

up_sam_outcome <- as.numeric(pred_up_sam > 0.5)
up_sam_tst <- as.numeric(pred_up_sam_t > 0.2)

confusionMatrix(as.factor(up_sam_tst), as.factor(shad_test_titan$Survived),
                                 positive = levels(as.factor(shad_test_titan$Survived))[2])





prop.table(table(shad_train_titan$Survived))
prop.table(table(shad_test_titan$Survived))


for_vif <- lm(Survived ~.-PassengerId -Embarked_C, data = shad_train_titan)

sort(vif(for_vif),decreasing = T)
# change target to factor
shad_train_titan$Survived <- as.factor(shad_train_titan$Survived)

#########################################################logistic regression#####################################
log_model <- glm(Survived ~.-PassengerId -Embarked_C -Embarked_S -Fare -Parch, data = shad_train_titan, family = "binomial") 

summary(log_model)

table(shad_train_titan$Survived)
shad_test_titan$pred_test <- predict(log_model, newdata = shad_test_titan, type = "response")
shad_train_titan$pred_train <- predict(log_model, newdata = shad_train_titan, type = "response")

library(pROC)
auc(roc(shad_test_titan$Survived, pred_test))#.8198
auc(roc(shad_train_titan$Survived, pred_train))#.8574

shad_test_titan$test_hard_class <- as.numeric(pred_test > 0.01)
confusionMatrix(as.factor(shad_test_titan$test_hard_class), as.factor(shad_test_titan$Survived),
                positive = levels(as.factor(shad_test_titan$Survived))[2])

library(randomForest)
####################################################build random forest##########################################
#######do parimeter optimization- Cross valivation

library(cvTools)

param <- list(mtry = c(5,10,15,20,25,35),
              ntree = c(50,100,200,500,700),
              maxnodes = c(5,10,15,20,30,50,100),
              nodesize = c(1,2,5,10))

mycost_auc <- function(y, yhat){
  roccurve = pROC::roc(y, yhat)
  score = pROC::auc(roccurve)
  return(score)
}

num_trials <- 120

subset_para <- function(full_list_para,n=10){
  all_comb = expand.grid(full_list_para)
  s = sample(1:nrow(all_comb),n)
  subset_para = all_comb[s,]
  return(subset_para)
}

my_params <- subset_para(param,num_trials)



myauc = 0


for (i in 1:num_trials){
  print(i)
  params_to_be_used = my_params[i,]
  ran_ref_fit <- cvTuning(randomForest,
                          Survived ~.-PassengerId -Embarked_C -Embarked_S -Fare -Parch,
                          data = shad_train_titan,
                          tuning = params_to_be_used,
                          folds = cvFolds(nrow(shad_train_titan),K=10,type = c("random")),
                          cost = mycost_auc, seed = 2,
                          predictArgs = list(type = c("prob")))
  current_auc = ran_ref_fit$cv[,2]
  if(current_auc > myauc){
    myauc = current_auc
    print(params_to_be_used)
    best_params = params_to_be_used
  }
}

#build model with best parameter gotten
shad_train_titan$Survived <- as.factor(shad_train_titan$Survived)

random_model <- randomForest(Survived ~.-PassengerId -Embarked_C -Embarked_S -Fare -Parch,
                             data = shad_train_titan, mtry = 15, ntree = 100, maxnodes = 20, nodesize = 2)



shad_pred_test <- predict(random_model, newdata = shad_test_titan, type = "prob")[,2]

auc(roc(shad_test_titan$Survived, shad_pred_test))
pROC::roc(shad_test_titan$Survived, shad_pred_test)

shad_pred_train <- predict(random_model, newdata = shad_train_titan, type = "prob")[,2]



library(ggplot2)

#ggplot(shad_train_titan, aes())

# get the proper cutoff


real <- shad_train_titan$Survived
#0.001, 0.999, 0.001

cutoffs <- seq(0.1,0.9,0.05)

cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(shad_pred_train>cutoff)
  
  TP =sum(real == 1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real == 1 & predicted == 0)
  
  
  
  P = TP + FN
  N = TN + FP
  
  Sn = TP/P
  Sp = TN/N
  precision = TP/(TP + FP)
  recall = Sn
  KS = (TP/P) - (FP/N)
  F5 =(26*precision*recall)/((25*precision) + recall)
  F.1 =(1.01*precision*recall)/((.01*precision) + recall)
  M = (4*FP+FN)/(5*(P+N))
  cutoff_data = rbind(cutoff_data,
                      c(cutoff,Sn,Sp,KS,F5,F.1,M))
  
}

cutoff_data = cutoff_data[-1,]

library(ggplot2)
prop.table(table(shad_train_titan$Survived))

library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

ggplot(cutoff_data, aes(x =cutoff, y = KS)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff



train_pred_outcome <- as.numeric(shad_pred_train > 0.20)

confusionMatrix(as.factor(train_pred_outcome), as.factor(shad_train_titan$Survived),
                positive = levels(as.factor(shad_train_titan$Survived))[2])


test_pred_outcome <- as.numeric(shad_pred_test > 0.20)

confusionMatrix(as.factor(test_pred_outcome), as.factor(shad_test_titan$Survived),
                positive = levels(as.factor(shad_test_titan$Survived))[2])

# treatment of imbalance data set
u_sample_d <- upSample(x = shad_train_titan[,colnames(shad_train_titan) != "Survived"],
                       y = as.factor(shad_train_titan$Survived))



#cutoffs <- seq(0.001,0.999,0.001)
#accuracy <- NULL

#for (i in seq(along = cutoffs)){
 # prediction <- ifelse(random_model$fitted.values >= cutoffs[i],1,0)
  #total_correct = length(which(shad_train_titan$Survived == prediction))
  #percentage_correct = total_correct/length(prediction)
  #accuracy <- c(accuracy,percentage_correct)
#}

#plot(cutoffs,accuracy,pch=19,type='b',xlab = "Cutoffs",ylab = "Accuracy%")

#################### build XGBOOST MODEL #################################################







library(xgboost)
xtrain <- train_data_titan %>% select(-Survived)
ytrain <- train_data_titan$Survived
ytrain <- as.numeric(as.factor(ytrain)) - 1



xgboy_model <- xgboost(data = data.matrix(xtrain),
                       label = ytrain,
                       objective = "reg:logistic",
                       eta =0.3,gamma =0,max_depth =6,min_child_weight=1,subsample=1,
                       colsample_bytree =1,nrounds = 100)



preddy_train <- predict(xgboy_model,newdata = data.matrix(xtrain))

preddy_test <- predict(xgboy_model,newdata = data.matrix(test_data_titan))

real <- ytrain
#0.001, 0.999, 0.001

cutoffs <- seq(0.1,0.9,0.05)

cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(preddy_train>cutoff)
  
  TP =sum(real == 1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real == 1 & predicted == 0)
  
  
  
  P = TP + FN
  N = TN + FP
  
  Sn = TP/P
  Sp = TN/N
  precision = TP/(TP + FP)
  recall = Sn
  KS = (TP/P) - (FP/N)
  F5 =(26*precision*recall)/((25*precision) + recall)
  F.1 =(1.01*precision*recall)/((.01*precision) + recall)
  M = (4*FP+FN)/(5*(P+N))
  cutoff_data = rbind(cutoff_data,
                      c(cutoff,Sn,Sp,KS,F5,F.1,M))
  
}

cutoff_data = cutoff_data[-1,]

library(ggplot2)
prop.table(table(shad_train_titan$Survived))

library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

ggplot(cutoff_data, aes(x =cutoff, y = KS)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff



preddy_outcome <- as.numeric(preddy_test > 0.4)

write.csv(preddy_outcome, "xgboostb4tun.csv", row.names = F)








