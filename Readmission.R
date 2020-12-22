library(ggplot2)
library(naniar)
library(gridExtra)
library(tidyverse)
library(caret)
library(ROSE)
library(e1071)
library(kernlab)
library(fastDummies)
library(data.table)
library(tree)
library(rattle)
library(randomForest)
library(factoextra)
library(fastDummies)


dia <- read.csv('diabetic_data.csv',na.strings=c("?"))

#############################  Data Cleaning  #############################  

# Null values
missing <- sapply(dia, function(x) round(sum(is.na(x))/length(x),3))
gg_miss_var(dia[,names(missing[missing>0])], 
            show_pct = T) + ggtitle('Percent rows missing')


dia <- dia[,-which(names(dia) == 'weight')]  # Drop weight
dia <- dia[,-which(names(dia) == 'payer_code')]  #Drop payer_code
dia <- dia[,-which(names(dia) == 'medical_specialty')] #Drop medical_specialty

# We will keep only the first encounter, as done in reference paper

dia = dia[order(dia[,'patient_nbr'],-dia[,'encounter_id']),]
uniq <- dia[!duplicated(dia$patient_nbr), ]


# Drop patients who either died or went to hospice
uniq <- uniq[-which(uniq$discharge_disposition_id %in% c(11,13,14,19,20,21)),]
# Drop remaining rows with nulls (~2% of data or less for race, diag_3, diag_2)
uniq <- uniq[complete.cases(uniq), ]  

###########################  Feature Engineering  ###########################  

# Recode diagnoses columns 
uniq$diag_1 <- case_when(
  uniq$diag_1 >= 390 & uniq$diag_1 <= 459 | uniq$diag_1 == 785 ~ 'Circulatory',
  uniq$diag_1 >= 460 & uniq$diag_1 <= 519 | uniq$diag_1 == 786 ~ 'Respiratory',
  uniq$diag_1 >= 520 & uniq$diag_1 <= 579 | uniq$diag_1 == 787 ~ 'Digestive',
  uniq$diag_1 >= 250 & uniq$diag_1 <= 251                     ~ 'Diabetes',
  uniq$diag_1 >= 800 & uniq$diag_1 <= 999                     ~ 'Injury',
  uniq$diag_1 >= 710 & uniq$diag_1 <= 739                     ~ 'Musculoskeletal',
  uniq$diag_1 >= 580 & uniq$diag_1 <= 629 | uniq$diag_1 == 788 ~ 'Genitourinary',
  uniq$diag_1 >= 390 & uniq$diag_1 <= 459 | uniq$diag_1 == 785 ~ 'Neoplasms',
  NA ~ as.character(uniq$diag_1),
)
uniq$diag_1 <- replace_na(uniq$diag_1,'Other')
table(uniq$diag_1)

uniq$diag_2 <- case_when(
  uniq$diag_2 >= 390 & uniq$diag_2 <= 459 | uniq$diag_2 == 785 ~ 'Circulatory',
  uniq$diag_2 >= 460 & uniq$diag_2 <= 519 | uniq$diag_2 == 786 ~ 'Respiratory',
  uniq$diag_2 >= 520 & uniq$diag_2 <= 579 | uniq$diag_2 == 787 ~ 'Digestive',
  uniq$diag_2 >= 250 & uniq$diag_2 <= 251                     ~ 'Diabetes',
  uniq$diag_2 >= 800 & uniq$diag_2 <= 999                     ~ 'Injury',
  uniq$diag_2 >= 710 & uniq$diag_2 <= 739                     ~ 'Musculoskeletal',
  uniq$diag_2 >= 580 & uniq$diag_2 <= 629 | uniq$diag_2 == 788 ~ 'Genitourinary',
  uniq$diag_2 >= 390 & uniq$diag_2 <= 459 | uniq$diag_2 == 785 ~ 'Neoplasms',
  NA ~ as.character(uniq$diag_2),
)
uniq$diag_2 <- replace_na(uniq$diag_2,'Other')
table(uniq$diag_2)

uniq$diag_3 <- case_when(
  uniq$diag_3 >= 390 & uniq$diag_3 <= 459 | uniq$diag_3 == 785 ~ 'Circulatory',
  uniq$diag_3 >= 460 & uniq$diag_3 <= 519 | uniq$diag_3 == 786 ~ 'Respiratory',
  uniq$diag_3 >= 520 & uniq$diag_3 <= 579 | uniq$diag_3 == 787 ~ 'Digestive',
  uniq$diag_3 >= 250 & uniq$diag_3 <= 251                     ~ 'Diabetes',
  uniq$diag_3 >= 800 & uniq$diag_3 <= 999                     ~ 'Injury',
  uniq$diag_3 >= 710 & uniq$diag_3 <= 739                     ~ 'Musculoskeletal',
  uniq$diag_3 >= 580 & uniq$diag_3 <= 629 | uniq$diag_3 == 788 ~ 'Genitourinary',
  uniq$diag_3 >= 390 & uniq$diag_3 <= 459 | uniq$diag_3 == 785 ~ 'Neoplasms',
  NA ~ as.character(uniq$diag_3),
)
uniq$diag_3 <- replace_na(uniq$diag_3,'Other')
table(uniq$diag_3)


# Recode discharge disposition
uniq$discharge_disposition_id <- case_when(
  uniq$discharge_disposition_id == 1 | uniq$discharge_disposition_id == 8  ~ 'Home'
)
uniq$discharge_disposition_id <- replace_na(uniq$discharge_disposition_id,'Other')
table(uniq$discharge_disposition_id)
# Recode admission source
uniq$admission_source_id <- case_when(
  uniq$admission_source_id == 7  ~ 'Emergency Room',
  uniq$admission_source_id == 1 | uniq$admission_source_id == 2 ~ 'Referral'
)
uniq$admission_source_id <- replace_na(uniq$admission_source_id,'Other')
table(uniq$admission_source_id)

# Combine number of visits
uniq$visits <- uniq$number_outpatient + uniq$number_emergency + uniq$number_inpatient
uniq <- uniq[,-c(which(names(uniq)=='number_outpatient'),which(names(uniq)=='number_inpatient'),
                 which(names(uniq)=='number_emergency'))]


# Combine number of procedures
uniq$procedures <- uniq$num_lab_procedures + uniq$num_procedures
uniq <- uniq[,-c(which(names(uniq)=='num_lab_procedures'),which(names(uniq)=='num_procedures'))]
# Drop enocunter_id and patient_nbr
uniq2 <- uniq[,-c(1,2)]  


# Dealing with medicines:
# Many columns have singular values
for (i in 15:37){
  print(cat(names(uniq2)[i],i))
  print(table(uniq2[,i]))
}

# If a column has at least 75 points in all four, keep
# If a column has a total of 1000 points not in "No", then convert to "Yes"
# Drop otherwise
dropped <- c(34:37,33,31,30,29,28,27,26,23,20,18,17,16)
full <- c(15,19,21,22,24,32)
two <- c(25)

for (col in two){
  uniq2[,col]<- case_when(
    uniq2[,col] == 'Steady' |  uniq2[,col] == 'Down' |  uniq2[,col] == 'Up' ~ 'Yes',
    T ~ as.character(uniq2[,col])
  )
}

uniq2 <- uniq2[,-dropped]

uniq2$age <- case_when(
  uniq2$age == '[10-20)' ~ '[0,30)',
  uniq2$age == '[0-10)' ~ '[0,30)',
  uniq2$age == '[20-30)' ~ '[0,30)',
  uniq2$age == '[30-40)' ~ '[30,60)',
  uniq2$age == '[40-50)' ~ '[30,60)',
  uniq2$age == '[50-60)' ~ '[30,60)'
)
uniq2$age <- replace_na(uniq2$age,'>60')
uniq2$age <- as.factor(uniq2$age)
table(uniq2$age)

# Recode target
uniq2$readmitted <- ifelse(uniq2$readmitted=='NO','No readmission','Readmission')
uniq2$readmitted <- as.factor(uniq2$readmitted)
table(uniq2$readmitted)
# Move readmitted to end of dataframe
uniq2 <- subset(uniq2, select = c(1:23,25:26,24))
# Get rid of 1 outlier in gender
uniq2 <- uniq2[-which(uniq2$gender=='Unknown/Invalid'),]
# Recode admission_type_id
uniq2$admission_type_id <- case_when(
  uniq2$admission_type_id == 1 | uniq2$admission_type_id == 2 ~ 'Emergency/Urgent',
  uniq2$admission_type_id == 3 ~ 'Elective'
)
uniq2$admission_type_id <- replace_na(uniq2$admission_type_id,'Other')
uniq2$admission_type_id <- as.factor(uniq2$admission_type_id)
table(uniq2$admission_type_id)
# Convert all characters to factors
for (i in names(uniq2[,colnames(Filter(is.character,uniq2))])){
  uniq2[,i] <- as.factor(uniq2[,i])
}
str(uniq2)
# Now we have 66192 rows, 26 variables



######################### Logistic regression #########################

set.seed(5)
# First, we split into train/test
train=sample(nrow(uniq2),0.75*nrow(uniq2))
train_data = uniq2[train,]
test_data = uniq2[-train,]
# Next, build a full model and perform backward selection to obtain a good set of variables
lr_base <- glm(readmitted~1,data=train_data,family='binomial')
lr_all = glm(readmitted~.,data=train_data,family='binomial')
backward <- step(lr_all, direction = 'backward', scope = list(lower=lr_base, 
                                                              upper = lr_all), trace = 1)
summary(backward)
# Now predict the model on the test set
prob = predict(backward,newdata = test_data, type= 'response')
prediction = rep('No readmission',length(prob))
prediction[prob>.5] = 'Readmission'
lr_conf <- confusionMatrix(data = as.factor(prediction), reference = as.factor(test_data$readmitted))
lr_conf$table
lr_conf 
lr_roc <- roc.curve(test_data$readmitted,prediction,plotit = T)
lr_roc
# Now reshuffle the data and use undersampling 
set.seed(5)
# High number of training data to compensate for undersampling deletion
train_under=sample(nrow(uniq2),0.85*nrow(uniq2))  
train_under_data = uniq2[train_under,]
test_under_data = uniq2[-train_under,]
# Use undersampling
lr_under <- ovun.sample(readmitted~.,data=train_data,method = 'under',N=table(train_data$readmitted)[2]*2)$data # New training set
table(lr_under$readmitted) # Check distribution
# rerun the model
lr_under_model <- glm(backward$formula,data=lr_under, family = 'binomial')
prob_under <- predict(lr_under_model,newdata = test_under_data, type= 'response')
prediction_under <- rep('No readmission',length(prob_under))
prediction_under[prob_under>.5] = 'Readmission'
# Check results
lr_conf_under <- confusionMatrix(data = as.factor(prediction_under), reference = as.factor(test_under_data$readmitted))
lr_conf_under$table
lr_conf_under 
lr_roc_under <- roc.curve(test_under_data$readmitted,prediction_under,plotit = T)
lr_roc_under
summary(backward)


################################ SVM #################################

# First we will split into train/test
svm_split <- sample(nrow(uniq2),0.65*nrow(uniq2))
svm_train = uniq2[svm_split,]
svm_test = uniq2[-svm_split,]
# Now oversample the training set
svm_train <- ovun.sample(readmitted~.,data=svm_train,method = 'over',N=table(svm_train$readmitted)[1]*2)$data # New training set
table(svm_train$readmitted)
# Now split into train/validation for hyperparameters
svm_val_split <- sample(nrow(svm_train),nrow(svm_train)*0.66)
svm_train <- svm_train[svm_val_split,]
svm_val <- svm_train[-svm_val_split,]

# Now we use RFE on training set to find the most important features
# Need only numeric columns for RFE function
dummy <- dummy_cols(svm_train,remove_selected_columns = T)
dummy <- dummy[,-c(86,87)]  
dummy <- as.data.frame(dummy)
dummy[,1:85] <- lapply(dummy, as.numeric) 
dummy <- as.data.frame(dummy)
dummy$readmitted <- svm_train$readmitted
dummy$readmitted <- ifelse(dummy$readmitted == 'No readmission','neg','pos')
dummy$readmitted <- factor(dummy$readmitted)

control <- rfeControl(functions=caretFuncs, method="cv", number=5,verbose=T)
# run the RFE 
svm_rfe <- rfe(dummy[,1:85], dummy[,86], sizes=c(1,3,5,10,20), rfeControl=control,method = 'svmRadial')
# summarize the results
print(svm_rfe)
# list the chosen features
predictors(svm_rfe)
# plot the results
plot(svm_rfe, type=c("g", "o"))
# Now we will use a validation set to tune hyperparameters
tune_control <- tune.control(sampling='fix')
tuned = tune(svm,readmitted~visits+number_diagnoses+procedures+time_in_hospital+
               diag_1+gender+diag_2+diag_3+discharge_disposition_id+admission_source_id+diabetesMed+      
               age+metformin,data=svm_train, tunecontrol = tune_control, 
             validation.x = svm_val[,c(1:25)],validation.y = svm_val[,26],kernel='radial',
             ranges =list(cost=2^(seq(-2,7,length.out = 10)),gamma=2^(seq(-8,-1,length.out = 8))))
summary(tuned)
tuned$best.parameters
best_svm <- tuned$best.model
# Now use tuned model to predict the test set
svm_pred <- predict(best_svm,svm_test,type='class')
svm_conf <- confusionMatrix(data = as.factor(svm_pred), reference = as.factor(svm_test$readmitted))
svm_conf
smv_roc <- roc.curve(svm_test$readmitted,svm_pred,plotit = T)  
smv_roc


############################ Decision Tree ############################

# Split into train/test
set.seed(5)
split <- sample(nrow(uniq2),nrow(uniq2)*0.65)
tree_train <- uniq2[split,]
tree_train <- ovun.sample(readmitted~.,data=tree_train,method = 'over',N=table(tree_train$readmitted)[1]*2)$data 
table(tree_train$readmitted)
tree_test <- uniq2[-split,]
# Using train to find the best value for cp
tree_caret <- train(backward$formula,data=tree_train,method='rpart',
                    trControl=trainControl(method='cv',number=10))
plot(tree_caret)
tree_caret$bestTune

fancyRpartPlot(tree_caret$finalModel)
caret_pred <- predict(tree_caret,tree_test)
tree_conf <- confusionMatrix(data = as.factor(caret_pred), reference = as.factor(tree_test$readmitted))
tree_conf$table
tree_conf
tree_roc <- roc.curve(tree_test$readmitted,tree_prob,plotit = T)
tree_roc

######################## Random Forest ##############################

set.seed(5)
train <- sample(nrow(uniq2),0.65*nrow(uniq2))
rf_train_data <- uniq2[train,]
rf_test_data <- uniq2[-train,]



rf_train_data <- ovun.sample(readmitted~.,data=rf_train_data,method = 'over',N=table(rf_train_data$readmitted)[1]*2)$data # New training set
table(rf_train_data$readmitted)

# Random Search
rf_control <- trainControl(method="cv", number=5, search="random")
mtry <- 5
rf_random <- train(readmitted~., data=rf_train_data, method="rf", metric='Accuracy', tuneLength=5, trControl=rf_control)
print(rf_random)
plot(rf_random)
rf_pred <- predict.train(rf_random,rf_test_data,type = 'raw')
rf_conf <- confusionMatrix(data = as.factor(rf_pred), reference = as.factor(rf_test_data$readmitted))
rf_conf$table
rf_conf
rf_roc <- roc.curve(rf_test_data$readmitted,rf_pred,plotit = T)
rf_roc

########################## K-means Clustering ##########################

cluster_y <- ifelse(uniq2$readmitted=='No readmission',1,2)
dummy <- dummy_cols(uniq2[,-26],remove_selected_columns = T)
dummy <- scale(dummy)
dummy <- as.data.frame(dummy)
km=kmeans(dummy,2,nstart=25)
table(km$cluster,cluster_y)
fviz_cluster(km,dummy)
wss <- function(k) {
  kmeans(dummy, k, nstart = 10)$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 1-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")


wss(4)