
#Load required  packages
library(cvTools)
library(pROC)
library(dplyr)
library(car)
library(caret)
library(randomForest)

# set up working directory
setwd("~/Documents/competition_data")
?read.csv
train_camp <- read.csv("bank-full_train.csv", stringsAsFactors = F)

test_camp <- read.csv("bank-full_test.csv", stringsAsFactors = F)

unique(train_camp$y)
train_camp <- train_camp %>% mutate(y = ifelse(y == "yes",1,0),
                                    y = as.numeric(y))
# make the columns uniform
test_camp$y <- NA
test_camp$data <- "test"
train_camp$data <- "train"

# combine test and train data set for data preprocessing

camp_data <- rbind(test_camp,train_camp)
unique(camp_data$pdays)
str(camp_data)
unique(camp_data$job)
?table
sort(prop.table(table(camp_data1$marital)))
# check the frequency rate with respect to the target variable
round(tapply(camp_data1$y, camp_data1$marital,function(x) mean(x, na.rm = TRUE)),2)

camp_data1 <- camp_data %>% mutate(job = ifelse(job %in% c("admin.","self-employed","unknown","entrepreneur","blue-collar","housemaid","management","technician","services","unemployed"),"blue-collar", job))
unique(camp_data1$job)

# treat marital variable
camp_data2 <- camp_data1 %>% mutate(marital = ifelse(marital %in% c("divorced","married"),"married", marital),
                                    marital = ifelse(marital == "married",1,0),
                                    marital = as.numeric(marital))

# treat marital variable
camp_data3 <- camp_data2 %>% mutate(education = ifelse(education %in% c("secondary","unknown"),"secondary", education))


camp_data4 <- camp_data3 %>% mutate(default = ifelse(default == "yes", 1,0),
                                    default = as.numeric(default))

camp_data5 <- camp_data4 %>% mutate(housing = ifelse(housing == "yes", 1,0),
                                    housing = as.numeric(housing))


camp_data6 <- camp_data5 %>% mutate(loan = ifelse(loan == "yes", 1,0),
                                    loan = as.numeric(loan))

camp_data7 <- camp_data6 %>% mutate(contact = ifelse(contact %in% c("unknown","cellular"), "cellular",contact),
                                    contact = ifelse(contact == "cellular",1,0),
                                    contact = as.numeric(contact))




camp_data8 <- camp_data7 %>% mutate(month = ifelse(month %in% c("jul","jun","jan","aug","may"), "may",month),
                                    month = ifelse(month %in% c("dec","oct","sep"),"oct", month),
                                    month = ifelse(month %in% c("apr","feb"),"apr", month))


camp_data9 <- camp_data8 %>% mutate(poutcome = ifelse(poutcome %in% c("unknown","failure"),"unknown", poutcome))



lapply(camp_data9, function(x) sum(is.na(x), na.rm = T))
str(camp_data2)

# convert categorical variable to numeric

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



for(var in c("job", "education", "month","poutcome")){
  camp_data10 = CreateDummies(camp_data9,var,100)
}

for(var in c("job", "education", "month")){
  camp_data11 = CreateDummies(camp_data10,var,100)
}
 
for(var in c("job", "education")){
  camp_data12 = CreateDummies(camp_data11,var,100)
}

for(var in c("job")){
  camp_data13 = CreateDummies(camp_data12,var,100)
}

glimpse(camp_data13) 

lapply(camp_data13, function(x) sum(is.na(x), na.rm = T))
prop.table(table(train_data$y))
# separate data into test and train
train_data <- camp_data13 %>% filter(data == "train") %>% select(-data)

test_data <- camp_data13 %>% filter(data == "test") %>% select(-y, -data)

# treat multicolinearity with linear model
for_vif <- lm(y~.-ID -month_may, data = train_data)

sort(vif(for_vif),decreasing = T)


# treat variable with improper p_value 
train_data$y <- as.factor(train_data$y)
log_reg = glm(formula = y ~ marital + balance + housing + loan + duration + 
                campaign + pdays + previous + poutcome_other + poutcome_unknown + 
                month_oct + month_apr + education_tertiary + education_secondary + 
                job_blue_collar -previous, family = "binomial", data = train_data)

summary(log_reg)

step(log_reg)

pred_train <- predict(log_reg, newdata = train_data, type = "response")

pROC::roc(train_data$y, pred_train)

train_outcome <- as.numeric(pred_train > 0.5)
library(InformationValue)
InformationValue::ks_stat(train_data$y,train_outcome)

confusionMatrix(as.factor(train_outcome), as.factor(train_data$y),
                positive = levels(as.factor(train_data$y))[2])



caret::confusionMatrix(as.factor(train_outcome), as.factor(train_data$y), positive = "1", mode = "everything")


#install.packages("InformationValue")

#  Class imbalance; Do upsampling
set.seed(41)
"%ni%" <- Negate("%in%")
up_train_samp <- upSample(x = train_data[, colnames(train_data) %ni% "y"],
                          y = train_data$y)
table(up_train_samp$Class)

# fit training data set to Logistic Regression

samp_train_log = glm(formula = Class ~ marital + balance + housing + loan + duration + 
                campaign + pdays + previous + poutcome_other + poutcome_unknown + 
                month_oct + month_apr + education_tertiary + education_secondary + 
                job_blue_collar -previous, family = "binomial", data = up_train_samp)


samp_pred_train <- predict(samp_train_log,newdata = up_train_samp, type = "response")

# check evaluation with AUC
pROC::roc(up_train_samp$Class, samp_pred_train)

#Use 0.5 as arbitrary cutoff
sam_pred_train_outcome <- as.numeric(samp_pred_train > 0.5)

#  Calculat Kilmogorov Smirnof-KS
InformationValue::ks_stat(up_train_samp$Class,sam_pred_train_outcome)
InformationValue::ks_plot(up_train_samp$Class,sam_pred_train_outcome)

#predict with test data set
sam_pred_test <- predict(samp_train_log, newdata = test_data, type = "response")

#get an  appropriate cutoff

real <- up_train_samp$Class
cutoffs <- seq(0.001,0.999,0.001)
cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(samp_pred_train>cutoff)
  TP =sum(real==1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real==1 & predicted == 0)
  
  
  
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
ggplot(cutoff_data, aes(x=cutoff,y=KS)) + geom_line()


library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff

# convert probability to  hard classes 
pred_test_outcome <- as.numeric(sam_pred_test > 0.4)

write.csv(pred_test_outcome, "Hedgar_Ajakaiye_P5_part2.csv", row.names = F)
################################################END OF LOGISTIC##############################

######################################### build random forest Model#################################
 
#RANDOM FOREST
#get best params for ranforest model
param <- list(mtry = c(5,10,15,20,25,35),
              ntree = c(50,100,200,500,700),
              maxnodes = c(5,10,15,20,30,50,100),
              nodesize = c(1,2,5,10))

mycost_auc <- function(y, yhat){
  roccurve = pROC::roc(y, yhat)
  score = pROC::auc(roccurve)
  return(score)
}

num_trials <- 84

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
                          formula = Class ~ marital + balance + housing + loan + duration + 
                            campaign + pdays + previous + poutcome_other + poutcome_unknown + 
                            month_oct + month_apr + education_tertiary + education_secondary + 
                            job_blue_collar -previous,
                          data = up_train_samp,
                          tuning = params_to_be_used,
                          folds = cvFolds(nrow(up_train_samp),K=10,type = c("random")),
                          cost = mycost_auc, seed = 2,
                          predictArgs = list(type = c("prob")))
  current_auc = ran_ref_fit$cv[,2]
  if(current_auc > myauc){
    myauc = current_auc
    print(params_to_be_used)
    best_params = params_to_be_used
  }
}


#Convert target class to factors
up_train_samp$Class <- as.factor(up_train_samp$Class)

#fit upsampled data set to random forest
random_f_fit <- randomForest(formula = Class ~ marital + balance + housing + loan + duration + 
                                   campaign + pdays + previous + poutcome_other + poutcome_unknown + 
                                   month_oct + month_apr + education_tertiary + education_secondary + 
                                   job_blue_collar -previous,data = up_train_samp,mtry = 5, ntree = 500, maxnodes = 100,nodesize =1)
                             

# get probability prediction
pred_test_samp <- predict(random_f_fit, newdata = test_data, type = "prob")[,2]

pred_train_samp <- predict(random_f_fit, newdata = up_train_samp, type = "prob")[,2]


# check to the best cutoff
real <- up_train_samp$Class
cutoffs <- seq(0.001,0.999,0.001)
cutoff_data <- data.frame(cutoff =99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  predicted <- as.numeric(pred_train_samp>cutoff)
  TP =sum(real==1 & predicted ==1)
  TN = sum(real == 0 & predicted == 0)
  FP = sum(real == 0 & predicted == 1)
  FN = sum(real==1 & predicted == 0)
  
  
  
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
ggplot(cutoff_data, aes(x=cutoff,y=KS)) + geom_line()


library(tidyr)
cutoff_long = cutoff_data %>%
  gather(Measure,Value,Sn:M)
ggplot(cutoff_long, aes(x=cutoff, y = Value, color=Measure)) + geom_line()

my_cutoff = cutoff_data$cutoff[which.max(cutoff_data$KS)]
my_cutoff


#Use cutoff to generate hard class
test_pred_outcome <- as.numeric(pred_test_samp > 0.522)

write.csv(test_pred_outcome,"Hedgar_Ajakaiye_P5_part2.csv", row.names = F )


samp_outcome <- as.numeric(samp_pred_train > 0.5)

# check the confusion matrix
caret::confusionMatrix(as.factor(samp_outcome), as.factor(up_train_samp$Class), positive = "1", mode = "everything")


library(InformationValue)

ks_stat(up_train_samp$Class, samp_outcome)


ks_plot(up_train_samp$Class, samp_outcome)

################################################ XGBOOST Algorithm ###########################################


