rm(list=ls(all=TRUE))

# Load libraries
library(tidyverse)
library(dplyr)
library(VIM)
library(ggplot2)
library(corrplot)
library(caret)
library(car)
library(mice)
library(glmnet)
library(pROC)

# Load the dataset
data <- read.csv("C:\\Users\\preci\\Downloads\\breast_cancer_data.csv")


#_________________________________________
# Exploratory Data Analysis
#_________________________________________

# Exploring the first few rows and structure of the dataset
nrow(data)
head(data) #output revealed some N/As
str(data)
summary(data)

# Check for missing data
# Summary of the number of missing values for each variable
missing_values <- sapply(data, function(x) sum(is.na(x)))
print(missing_values) 
sum(missing_values) # a total of 186 missing data points

missing_values_diagnosis <- is.na(data$diagnosis)
sum(missing_values_diagnosis) # no missing value in diagnosis

missing_values_diagnosis <- is.na(data$diagnosis_code)
sum(missing_values_diagnosis) # no missing values in diagnosis_code

# Drop the 'X', 'id', and 'diagnosis' columns 
# these will not be needed subsequently
# diagnosis has already been encoded as diagnosis_code
data_cleaned <- select(data, -c(X, id, diagnosis))

# Visualising the distribution of diagnosis
p <- ggplot(data_cleaned, aes(x = factor(diagnosis_code))) +
  geom_bar() +
  labs(x = 'Diagnosis', y = 'Count', title = 'Distribution of Diagnosis') +
  scale_x_discrete(labels = c('0' = 'Benign', '1' = 'Malignant'))

p + geom_text(stat = 'count', aes(label = ..count..), vjust = -0.5, 
              position = position_stack(vjust = 1.0))

# Assess pattern of missingness in the dataset
par(las=0) 
md.pattern(data_cleaned)

# Visualise missing data and explore the missingness pattern
mice_plot <- aggr(data_cleaned, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(data_cleaned), cex.axis=.7,
                  gap=3, ylab=c("Missing data", "Pattern"))

# Plotting a correlation matrix plot
data_numeric <- data_cleaned[sapply(data_cleaned, is.numeric)]
cor_matrix <- cor(data_numeric)
print(cor_matrix)

library(reshape2)
cor_data <- melt(cor_matrix)
ggplot(data = cor_data, aes(x = Var1, y = Var2, fill = value)) + 
  geom_tile() +
  scale_fill_gradient2(low = "red", high = "blue", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Assessing Multicollinearity in the dataset
logit_model <- glm(diagnosis_code ~ ., data=data_cleaned, family=binomial)
vif_value <- vif(logit_model)
print(vif_value)
plot(vif_value)


#___________________________________
# Model Development
#___________________________________

# Spitting the dataset into Train/Test sets (80:20)

set.seed(123)
trainIndex <- createDataPartition(data_cleaned$diagnosis_code, p = .8, 
                                  list = FALSE, 
                                  times = 1)

trainSet <- data_cleaned[trainIndex, ]
testSet <- data_cleaned[-trainIndex, ]

# Performing various methods of imputation on the train_set
#__________________________________________________________
# Logistic Regression Model Applying Complete Case Analysis 
#____________________________________________________________
# Methods of regularisation to be used in this model
# L1- Lasso (alpha = 1)
# L1+L2- Elastic Net (alpha = 0.5)
# L2- Ridge (alpha = 0)

# Applying Complete Case Analysis separately to the training and test sets
cc_trainSet <- na.omit(trainSet)
cc_testSet <- na.omit(testSet)

# Setting target variable 'diagnosis_code' as a factor in both sets
cc_trainSet$diagnosis_code <- as.factor(cc_trainSet$diagnosis_code)
cc_testSet$diagnosis_code <- as.factor(cc_testSet$diagnosis_code)

# Creating matrix of predictors and the response variable for glmnet
x_train <- model.matrix(diagnosis_code ~ . -1, data = cc_trainSet) 
y_train <- cc_trainSet$diagnosis_code

x_test <- model.matrix(diagnosis_code ~ . -1, data = cc_testSet)
y_test <- cc_testSet$diagnosis_code

# Fitting the logistic regression model with cross-validation
cv_fit <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, 
                    nfolds = 5)

# Plotting the lambda selection
plot(cv_fit)

# Using the lambda that gives minimum mean cross-validated error
best_lambda <- cv_fit$lambda.min

# Refitting the model on the selected lambda
fit <- glmnet(x_train, y_train, family = "binomial", alpha = 1, 
              lambda = best_lambda)

# Predicting on the test set
predictions_prob <- predict(fit, newx = x_test, type = "response")
predictions <- ifelse(predictions_prob > 0.5, "1", "0")

# Calculating the confusion matrix for the test set
confusionMatrix <- confusionMatrix(as.factor(predictions), y_test)
print(confusionMatrix)

# ROC Curve and AUC

# For the test set
roc_test <- roc(response = as.numeric(as.factor(y_test)), 
                predictor = as.numeric(predictions_prob))
plot(roc_test, main = "ROC Curve - Test Data")
auc_test <- auc(roc_test)
print(paste("AUC for Test Data:", auc_test))

# For the training set
predictions_prob_train <- predict(fit, newx = x_train, type = "response")
roc_train <- roc(response = as.numeric(as.factor(y_train)), 
                 predictor = as.numeric(predictions_prob_train))
plot(roc_train, main = "ROC Curve - Train Data")
auc_train <- auc(roc_train)
print(paste("AUC for Train Data:", auc_train))

#___________________________________________________________________________
# Logistic Regression Model Applying mice Mean Imputation for Missing Data 
#____________________________________________________________________________
# Methods of regularisation to be used in this model
# L1- Lasso (alpha = 1)
# L!+L2- Elastic Net (alpha = 0.5)
# L2- Ridge (alpha = 0)


# Imputing missing data in the training set using MICE
set.seed(123)
imputed_trainSet <- mice(trainSet, m=1, maxit=50, method='mean', seed=500)
completeTrainSet <- complete(imputed_trainSet)

# Imputation into the test set
imputed_testSet <- mice(testSet, m=1, maxit=50, method='mean', seed=500)
completeTestSet <- complete(imputed_testSet)

# Setting target variable 'diagnosis_code' as a factor in both imputed sets
completeTrainSet$diagnosis_code <- as.factor(completeTrainSet$diagnosis_code)
completeTestSet$diagnosis_code <- as.factor(completeTestSet$diagnosis_code)

# Creating matrix of predictors and the response variable for glmnet
x_train_imputed <- model.matrix(diagnosis_code ~ . -1, data = completeTrainSet) 
y_train_imputed <- completeTrainSet$diagnosis_code

x_test_imputed <- model.matrix(diagnosis_code ~ . -1, data = completeTestSet)
y_test_imputed <- completeTestSet$diagnosis_code

# Fitting the logistic regression model with cross-validation on imputed 
#data
cv_fit_imputed <- cv.glmnet(x_train_imputed, y_train_imputed, 
                            family = "binomial", alpha = 0.5, nfolds = 5)
plot(cv_fit_imputed)

# Using the lambda that gives minimum mean cross-validated error
best_lambda_imputed <- cv_fit_imputed$lambda.min

# Refitting the model on the selected lambda with imputed data
fit_imputed <- glmnet(x_train_imputed, y_train_imputed, family = "binomial", 
                      alpha = 0.5, lambda = best_lambda_imputed)

# Predicting on the imputed test set
predictions_prob_imputed <- predict(fit_imputed, newx = x_test_imputed, 
                                    type = "response")
predictions_imputed <- ifelse(predictions_prob_imputed > 0.5, "1", "0")

# Calculating the confusion matrix for the imputed test set
confusionMatrix_imputed <- confusionMatrix(as.factor(predictions_imputed), 
                                           y_test_imputed)
print(confusionMatrix_imputed)

# ROC Curve and AUC for imputed sets

# Test set
roc_test_imputed <- roc(response = as.numeric(as.factor(y_test_imputed)), 
                        predictor = as.numeric(predictions_prob_imputed))
plot(roc_test_imputed, main = "ROC Curve - Test Data with Imputation")
auc_test_imputed <- auc(roc_test_imputed)
print(paste("AUC for Test Data with Imputation:", auc_test_imputed))

# Training set
predictions_prob_train_imputed <- predict(fit_imputed, newx = x_train_imputed, 
                                          type = "response")
roc_train_imputed <- roc(response = as.numeric(as.factor(y_train_imputed)), 
                         predictor = as.numeric(predictions_prob_train_imputed))
plot(roc_train_imputed, main = "ROC Curve - Train Data with Imputation")
auc_train_imputed <- auc(roc_train_imputed)
print(paste("AUC for Train Data with Imputation:", auc_train_imputed))


#__________________________________________________________________________
# LOGISTIC REGRESSION MODEL APPLYING HOT DECK IMPUTATION FOR MISSING DATA 
#__________________________________________________________________________
# Methods of regularisation to be used in this model
# L1- Lasso (alpha = 1)
# L1+L2- Elastic Net (alpha = 0.5)
# L2- Ridge (alpha = 0)

# Imputing missing data in the training set using Hot Deck Imputation
set.seed(500)
imputed_Data_pmm <- mice(trainSet, m=5, maxit=50, method='pmm', seed=500)
completeTrain_pmm <- complete(imputed_Data_pmm, 2)

# Imputing the test set
imputed_testSet_pmm <- mice(testSet, m=5, maxit=50, method='pmm', seed=500)
completeTest_pmm <- complete(imputed_testSet_pmm, 2)

# Setting target variable 'diagnosis_code' as a factor in both completed sets
completeTrain_pmm$diagnosis_code <- as.factor(completeTrain_pmm$diagnosis_code)
completeTest_pmm$diagnosis_code <- as.factor(completeTest_pmm$diagnosis_code)

# Creating matrix of predictors and the response variable for glmnet
x_train_pmm <- model.matrix(diagnosis_code ~ . -1, data = completeTrain_pmm) 
y_train_pmm <- completeTrain_pmm$diagnosis_code

x_test_pmm <- model.matrix(diagnosis_code ~ . -1, data = completeTest_pmm)
y_test_pmm <- completeTest_pmm$diagnosis_code

# Fitting the logistic regression model with cross-validation
cv_fit_pmm <- cv.glmnet(x_train_pmm, y_train_pmm, family = "binomial", 
                        alpha = 0.5, nfolds = 5)

# Using the lambda that gives minimum mean cross-validated error
best_lambda_pmm <- cv_fit_pmm$lambda.min

# Refitting the model on the selected lambda with PMM imputed data
fit_pmm <- glmnet(x_train_pmm, y_train_pmm, family = "binomial", alpha = 0.5, 
                  lambda = best_lambda_pmm)

# Predicting on the PMM imputed test set
predictions_prob_pmm <- predict(fit_pmm, newx = x_test_pmm, type = "response")
predictions_pmm <- ifelse(predictions_prob_pmm > 0.5, "1", "0")

# Calculating the confusion matrix for the PMM imputed test set
confusionMatrix_pmm <- confusionMatrix(as.factor(predictions_pmm), y_test_pmm)
print(confusionMatrix_pmm)

# ROC Curve and AUC for PMM imputed sets

# Test set
roc_test_pmm <- roc(response = as.numeric(as.factor(y_test_pmm)), 
                    predictor = as.numeric(predictions_prob_pmm))
plot(roc_test_pmm, main = "ROC Curve - Test Data with PMM Imputation")
auc_test_pmm <- auc(roc_test_pmm)
print(paste("AUC for Test Data with PMM Imputation:", auc_test_pmm))

# Training set
predictions_prob_train_pmm <- predict(fit_pmm, newx = x_train_pmm, 
                                      type = "response")
roc_train_pmm <- roc(response = as.numeric(as.factor(y_train_pmm)), 
                     predictor = as.numeric(predictions_prob_train_pmm))

plot(roc_train_pmm, main = "ROC Curve - Train Data with PMM Imputation")
auc_train_pmm <- auc(roc_train_pmm)
print(paste("AUC for Train Data with PMM Imputation:", auc_train_pmm))


#_____________________________________________________________________________
# Logistic Regression Model Applying Linear Regression Prediction-Based 
# Imputation
#______________________________________________________________________________
# Methods of regularisation to be used in this model
# L1- Lasso (alpha = 1)
# L1+L2- Elastic Net (alpha = 0.5)
# L2- Ridge (alpha = 0)

# Training set
set.seed(500)
imputed_Data_norm <- mice(trainSet, m=5, maxit=50, method='norm', seed=500)
completeTrain_norm <- complete(imputed_Data_norm, 2)

# Preparing the test set
imputed_testSet_norm <- mice(testSet, m=5, maxit=50, method='norm', seed=500)
completeTest_norm <- complete(imputed_testSet_norm, 2)

# Setting target variable 'diagnosis_code' as a factor in both completed sets
completeTrain_norm$diagnosis_code <- as.factor(completeTrain_norm$diagnosis_code)
completeTest_norm$diagnosis_code <- as.factor(completeTest_norm$diagnosis_code)

# Creating matrix of predictors and the response variable for glmnet
x_train_norm <- model.matrix(diagnosis_code ~ . -1, data = completeTrain_norm) 
y_train_norm <- completeTrain_norm$diagnosis_code

x_test_norm <- model.matrix(diagnosis_code ~ . -1, data = completeTest_norm)
y_test_norm <- completeTest_norm$diagnosis_code

# Fitting the logistic regression model with cross-validation on Norm 
#imputed data
cv_fit_norm <- cv.glmnet(x_train_norm, y_train_norm, family = "binomial", 
                         alpha = 0.5, nfolds = 5)

# Using the lambda that gives minimum mean cross-validated error
best_lambda_norm <- cv_fit_norm$lambda.min

# Refitting the model on the selected lambda with Norm imputed data
fit_norm <- glmnet(x_train_norm, y_train_norm, family = "binomial", alpha = 0.5, 
                   lambda = best_lambda_norm)

# Predicting on the Norm imputed test set
predictions_prob_norm <- predict(fit_norm, newx = x_test_norm, 
                                 type = "response")
predictions_norm <- ifelse(predictions_prob_norm > 0.5, "1", "0")

# Calculating the confusion matrix for the Norm imputed test set
confusionMatrix_norm <- confusionMatrix(as.factor(predictions_norm), y_test_norm)
print(confusionMatrix_norm)

# ROC Curve and AUC for the Test set with Normal Imputation
roc_test_norm <- roc(response = as.numeric(as.factor(y_test_norm)), 
                     predictor = as.numeric(predictions_prob_norm))

plot(roc_test_norm, main = "ROC Curve - Test Data with Normal Imputation")
auc_test_norm <- auc(roc_test_norm)
print(paste("AUC for Test Data with Normal Imputation:", auc_test_norm))

# ROC Curve and AUC for the Training set with Normal Imputation
predictions_prob_train_norm <- predict(fit_norm, newx = x_train_norm, 
                                       type = "response")

roc_train_norm <- roc(response = as.numeric(as.factor(y_train_norm)), 
                      predictor = as.numeric(predictions_prob_train_norm))

plot(roc_train_norm, main = "ROC Curve - Train Data with Normal Imputation")
auc_train_norm <- auc(roc_train_norm)
print(paste("AUC for Train Data with Normal Imputation:", auc_train_norm))





