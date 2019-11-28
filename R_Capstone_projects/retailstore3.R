#Load necessary libraries
library(dplyr)
library(pROC)
library(cvTools)
library(caret)
library(car)
library(randomForest)
library(ggplot2)
library(xgboost)

# set up working directory
setwd("~/Documents/competition_data")
train_store <- read.csv("store_train-1.csv", stringsAsFactors = F)
test_store <- read.csv("store_test-1.csv", stringsAsFactors = F)
dim(test_store)

# combine test and train data set so as to uniformly carryout data preprocessing

# harmonize rows with column first for easy separation after preprocessing
test_store$store <- NA
test_store$data <- "test"
train_store$data <- "train"


super_store <- rbind(test_store, train_store)
unique(super_store$countyname)

lapply(super_store, function(x) sum(is.na(x)))
?is.na
# Distribution of store types in California

# remove the two row with missing values of population

#super_store1 <- super_store %>% select("Id","sales0","sales1","sales2","sales3","sales4",
                     #  "country","State","CouSub","countyname","storecode",
                       #"Areaname","countytownname","population","state_alpha",
                       #"store_Type","store","data") %>% filter(!is.na(population))


# remove the row with country missing

#super_store2 <- super_store1 %>% select("Id","sales0","sales1","sales2","sales3","sales4",
                               #        "country","State","CouSub","countyname","storecode",
                                 #      "Areaname","countytownname","population","state_alpha",
                                   #    "store_Type","store","data") %>% filter(!is.na(country))


lapply(super_store, function(x) sum(is.na(x)))



# let go of the following columes as their percentage of frequency is not significant
### State, CouSub, storecode,Areaname, countytownname,countyname

super_store1 <- super_store %>% select(-country, -Areaname, -countyname,-storecode,-CouSub,-countytownname,-State,-state_alpha)



#sort(table(super_store3$state_alpha), decreasing = T)

#sort(round(tapply(super_store3$store, super_store3$store_Type, function(x) mean(x,na.rm = T)),1))


# let go of the
#super_store2 <- super_store1 %>% mutate(state_alpha = ifelse(state_alpha %in% c("GU","VI"), "VI", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("ND","WY"),"ND", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("MT","NE","NV","VT",""),"VT", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("AK","AR","CO","IA","ID","KS","KY","MN","NC","OK","SD","WA"),"KY", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("AL","AZ","GA","ME","MI","MO","MS","NH","OH","OR","PA","SC","TX","UT","WI"),"ME", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("IL","IN","LA","VA","WV"),"IN", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("CA","FL","HI","MD","NY","TN"),"TN", state_alpha),
                                       # state_alpha = ifelse(state_alpha %in% c("CT","DE"),"CT", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("PR","RI"),"PR", state_alpha),
                                        #state_alpha = ifelse(state_alpha %in% c("MA","NJ"),"MA", state_alpha))



#super_store3 <- super_store1 %>% mutate(store_Type = ifelse(store_Type %in% c("Grocery Store","Supermarket Type3"), "Grocery Store", store_Type),
                                       # store_Type = ifelse(store_Type %in% c("Supermarket Type1","Supermarket Type2"),"Supermarket Type1", store_Type),
                                        #store_Type = ifelse(store_Type == "Grocery Store", 1,0),
                                        #store_Type = as.numeric(store_Type))

table(super_store1$store_Type)

View(super_store1)

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

super_store2 <- CreateDummies(super_store1,"store_Type",100)

glimpse(super_store2)
lapply(super_store2, function(x) sum(is.na(x)))

#replace missing 
mean(super_store4$population, na.rm = T)

# detect and treat missing values
#all_house_data3 <- a
#super_store4[!((is.na(super_store4$store)) & super_store4$data== "train"),]

super_store3 <- super_store2

for(col in names(super_store3)){
  if(sum(is.na(super_store3[,col])) > 0 & !(col %in% c("data", "store"))){
    super_store3[is.na(super_store3[,col]), col] = median(super_store3[super_store3$data == "train",col],na.rm = T)
  }
  
}

lapply(super_store3, function(x) sum(is.na(x)))



super_train <- super_store3 %>% filter(data == "train") %>% select(-data)
super_test <- super_store3 %>% filter(data == "test" ) %>% select(-store,-data) 

dim(super_train)
dim(super_test)

lapply(super_test, function(x) sum(is.na(x), na.rm = T))
boxplot(super_train)
# take care of multicolinearity

for_vif <- lm(store~.-Id -sales0 -sales2 -sales3 -sales4, data = super_train)

sort(vif(for_vif), decreasing = T)

#############################################Build logistic regression model##############################
super_train$store <- as.factor(super_train$store)
log_reg_model <- glm(formula = store ~ sales1 + population, family = "binomial", 
                     data = super_train)

summary(log_reg_model)

step(log_reg_model)


prop.table(table(super_train$store))

train_pred <- predict(log_reg_model, newdata = super_train, type = "response")

test_pred <- predict(log_reg_model, newdata = super_test, type = "response")

write.csv(test_pred,"Hedgar_Ajakaiye_P2_part2.csv", row.names = F )

pROC::roc(super_train$store, train_pred)#75/74

train_outcome <- as.numeric(train_pred > 0.5)

caret::confusionMatrix(as.factor(train_outcome), as.factor(super_train$store), positive = "1", mode = "everything")

############################################### Build RANDOM FOREST MODEL#################################

# Get appropriate parameters for random forest


param <- list(mtry = c(5,10,15,20,25,35),
              ntree = c(700, 800,900,1200,1500,2000),
              maxnodes = c(100,120,150,190,230,300),
              nodesize = c(2,5,10, 20,50,100))

mycost_auc <- function(y, yhat){
  roccurve = pROC::roc(y, yhat)
  score = pROC::auc(roccurve)
  return(score)
}

num_trials <- 130

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
                          formula = store ~ sales1 + population,
                          data = super_train,
                          tuning = params_to_be_used,
                          folds = cvFolds(nrow(super_train),K=10,type = c("random")),
                          cost = mycost_auc, seed = 2,
                          predictArgs = list(type = c("prob")))
  current_auc = ran_ref_fit$cv[,2]
  if(current_auc > myauc){
    myauc = current_auc
    print(params_to_be_used)
    best_params = params_to_be_used
  }
}


super_train$store <- as.factor(super_train$store)


######################################Random forest model###########################################
retail_rand_f <- randomForest(formula = store ~ sales1 + population,data = super_train,mtry = 5, ntree = 2000, maxnodes = 300, nodesize = 20)



preddy_train <- predict(retail_rand_f, newdata = super_train, type = "prob")[,2]

pROC::roc(super_train$store, preddy_train)
pROC::auc(super_train$store, preddy_train)
preddy_test <- predict(retail_rand_f, newdata = super_test, type = "prob")[,2]

write.csv(preddy_test, "Hedgar_Ajakaiye_P2_part2.csv", row.names = F)


#####################################Decision tree Model#########################################
library(rpart)
install.packages("rpart.plot")
library(rpart.plot)
# decision tree model
decision_tree <- rpart(formula = store ~ sales1 + population,method = "class", data = super_train)

printcp(decision_tree)
plotcp(decision_tree)
rpart.plot(decision_tree)

pred_dec <- predict(decision_tree, newdata = super_train, type = "prob")[,2]

pROC::roc(super_train$store, pred_dec)



############################## XBBOOST MODEL ######################################################



install.packages("gbm")

install.packages("xgboost")
library(xgboost)
xtrain <- super_train %>% select(-store)
ytrain <- super_train$store
ytrain <- as.numeric(as.factor(ytrain)) - 1



xgboy_model <- xgboost(data = data.matrix(xtrain),
                      label = ytrain,
                      objective = "reg:logistic",
                      eta =0.3,gamma =0,max_depth =6,min_child_weight=1,subsample=1,
                      colsample_bytree =1,nrounds = 100)



print(xgboy_model)

xgpred <- predict(xgboy_model,newdata = data.matrix(xtrain))

xgpred_test <- predict(xgboy_model, data.matrix(super_test))

write.csv(xgpred_test, "Hedgar_Ajakaiye_P2_part2.csv", row.names = F)

pROC::roc(ytrain, xgpred)











