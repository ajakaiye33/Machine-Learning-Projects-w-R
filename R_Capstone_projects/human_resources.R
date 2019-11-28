# Load necessary libraries
library(dplyr)
library(caret)
library(car)
library(ggplot2)
library(pROC)
library(ROCR)
library(tidyr)
library(cvTools)
library(randomForest)



#set up work splace

setwd("~/Documents/R_Eda_ML/machine_learning_projects")

hr_train_data <- read.csv("hr_train.csv", stringsAsFactors = FALSE)
hr_test_data <- read.csv("hr_test.csv", stringsAsFactors = FALSE)


#combine train and test for preprpcessing
hr_test_data$left = NA
hr_test_data$data <- "test"
hr_train_data$data <- "train"


com_hr_data <- rbind(hr_test_data, hr_train_data)

# Data preprocessing

sort(prop.table(table(com_hr_data$sales)))

round(tapply(com_hr_data$left,com_hr_data$sales, function(x) mean(x,na.rm = T)),2)

com_hr_data1 <- com_hr_data %>% mutate(sales = ifelse(sales %in% c("sales","support","IT","product_mng","RandD"),"sales",sales),
                                       sales = ifelse(sales %in% c("accounting", "hr", "marketing","technical"), "technical",sales),
                                       salary = ifelse(salary %in% c("low", "high"),"low", salary))

str(com_hr_data1)
#check for missing values
lapply(com_hr_data1, function(x) sum(is.na(x)))


#convert caregorical variable to numerical variable
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


for(var in c("salary", "sales")){
  com_hr_data2 = CreateDummies(com_hr_data1,var,50)
}


glimpse(com_hr_data2)

com_hr_data3 <- com_hr_data2 %>% mutate(salary = ifelse(salary == "low",1,0),
                                        salary = as.numeric(salary))

glimpse(com_hr_data3)

# check for missing value
lapply(com_hr_data3, function(x) sum(is.na(x)))

# Divide data into test and train

train_data_hr <- com_hr_data3 %>% filter(data == "train") %>% select(-data)
test_data_hr <- com_hr_data3 %>% filter(data == "test") %>% select(-data,-left)

# convert target to factor
train_data_hr$left <- as.factor(train_data_hr$left)


#fit to linear regression model for the purpose treating multicolinearity
for_vif <- lm(left ~.-sales_sales,data = train_data_hr)

sort(vif(for_vif), decreasing = T)

# fit to logitics regression model
log_model <- glm(left ~.-sales_sales,data = train_data_hr, family = "binomial")


summary(log_model)
step(log_model)

pred_train <- predict(log_model, newdata = train_data_hr, type = "response")

#check the AUC/ROC value
pROC::roc(train_data_hr$left, pred_train)

pred_test <- predict(log_model, newdata = test_data_hr, type = "response")
write.csv(pred_test,"prob_logistic_reg_prediction.csv", row.names = FALSE)