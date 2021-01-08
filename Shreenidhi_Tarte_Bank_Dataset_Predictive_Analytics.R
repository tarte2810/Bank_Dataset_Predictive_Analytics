#ALY6020 Final Group Project
#Bank Marketing Dataset
install.packages("e1071")

###Step 1 - Install and Load the required packages
library(lattice)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(C50)
library(gmodels)
library(corrplot)
library(randomForest)
library(ggpubr)
library(tidyverse)
library(MLmetrics)
library(DALEX)
library(e1071)

###Step 2 - Collecting data
#The marketing campaigns of a Portuguese banking institution dataset has been downloaded 
#from UCI Machine Learning Repository
#Link: https://archive.ics.uci.edu/ml/datasets/bank+marketing
#Read in the data
bank<-read.csv('bank_final.csv')
#Obtaining first several rows of the data
head(bank)


###Step 3 - Data exploring and preparing
#Observing the data structure
str(bank)
summary(bank)

#Checking for missing values
anyNA(bank)

### Step 4 - Performing Exploratory Data Analysis
#Extracting numerical variables by slicing
bank1<-bank[,c(1,6,10,12,13,14,15)]
#Implementing cor() to generate correlation matrix
bank2<-cor(bank1)

#Plotting visualization
#EDA1 - Correlation plot
a<-corrplot(bank2, type = "upper", order = "hclust",tl.col = "black", tl.srt = 45)
a

#EDA 2 - Distribution of duration of loan 
ggplot(bank, aes(duration)) + geom_histogram(binwidth = 50)+
  scale_x_continuous("duration")+
  scale_y_continuous("Count")+ labs(title = "Histogram")

#EDA 3 - Observing has the client subscribed a term deposit based on duration variable?
bank %>% ggplot(aes(x=duration, fill=y)) +
  geom_density(alpha=0.5) 

#EDA4 - Type of jobs in people
ggplot(bank, aes(job,balance)) + geom_bar(stat = "identity", fill = "darkblue")+
  scale_x_discrete("Job")+ scale_y_continuous("Item Weight")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5)) + labs(title = "Bar plot")

#EDA5 - Marital status in people
ggplot(bank, aes(x=marital)) +
  geom_bar(fill="steelblue")+
  ggtitle("Barplot showing marital status in people")+
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5),panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggpubr::rotate_x_text()

#EDA6 - Education status in people
ggplot(bank, aes(x=education)) +
  geom_bar(fill="pink")+
  ggtitle("Barplot showing education status in people")+
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5),panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggpubr::rotate_x_text()

#EDA7 - Distribution of loan default
counts <- table(bank$default)
bp <- barplot(counts,col=c("blue","red"), legend = rownames(counts), main = "Term Deposit")

#EDA8 - Distribution of output variable 
ggplot(bank, aes(x=y)) +
  geom_bar(fill="green")+
  ggtitle("Distribution of output variable - y ")+
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5),panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))+
  ggpubr::rotate_x_text()

#EDA9 - AGE DISTRIBUTION
gg = ggplot (bank) 
p1 = gg + geom_histogram(aes(x=age),color="black", fill="white", binwidth = 5) +
  ggtitle('Age Distribution (red mean line)') +
  ylab('Count') +
  xlab('Age') +
  geom_vline(aes(xintercept = mean(age), color = "red")) +
  scale_x_continuous(breaks = seq(0,100,5)) +
  theme(legend.position = "none")
p1

#EDA10 - Observing the client subscribed term deposit or not based on age.
bank %>% ggplot(aes(x=age, fill=y)) +
  geom_density(alpha=0.5) 

###Step 5 - Training models on the data
#First Model: Decision Tree
#Creating random train and test datasets
set.seed(123)

intrain <- createDataPartition(bank$y, p= 0.8, list = FALSE)
training <- bank[intrain,] #Training dataset contains 80% of random data
testing <- bank[-intrain,] #Testing dataset contains 20% of random data

#Checking the number of observations and variables in both datasets
dim(training)
dim(testing)

#Training the Decision Tree classifier with criterion as information gain
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
dtree_fit <- train(y ~., data = training, method = "rpart",
                   parms = list(split = "information"),
                   trControl=trctrl,
                   tuneLength = 10)

dtree_fit

#Visualizing the decision tree
prp(dtree_fit$finalModel, box.palette = "Reds", tweak = 1.2)

###Evaluating Model Performance
#Predicting the model on test data
test_pred <- predict(dtree_fit, newdata = testing)
confusionMatrix(test_pred, testing$y)

###Improving model performance
#Training the Decision Tree classifier with criterion as gini index
dtree_fit_gini <- train(y ~., data = training, method = "rpart",
                        parms = list(split = "gini"),
                        trControl=trctrl,
                        tuneLength = 10)
dtree_fit_gini

#Plotting the decison tree
prp(dtree_fit_gini$finalModel, box.palette = "Blues", tweak = 1.2)

###Evaluating model performance
#Predicting the model on test data
test_pred_gini <- predict(dtree_fit_gini, newdata = testing)
confusionMatrix(test_pred_gini, testing$y )

#Second Model: Random Forest 

rf <- randomForest(y~., data = training)
rf

#Evaluating variable Importance
importance(rf)
varImpPlot(rf)

###Evaluating model performance
#Predicting the accuracy of test data
test_result <- predict(rf,testing)
confusionMatrix(test_result, testing$y)
error <- mean(test_result != testing$y)
print(paste('Accuracy',1-error))

#ROC and AUC for the model
library(Metrics)
library(ROCR)
pr <- prediction(as.numeric(test_result), as.numeric(testing$y))
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
plot(prf, lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')

library(AUC)
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
auc

#Third Model: Logistic Regression
#Define model settings
set.seed(123)
#Splitting
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary) # return more metrics than binary classification

#Creating the classifier
set.seed(123) # for reproducibility
#Logistic Regression
model_glm <- train(y~.,
                   data = training,
                   method = "glm",
                   family = "binomial",
                   # preProcess = "pca",
                   trControl = control)

#Observing the GLM model
print(model_glm)

#Evaluate the Logistic Regression Prediction model
#Out-of-sample testing

#Actual predictions
pred_glm_raw <- predict.train(model_glm,
                              newdata = testing,
                              type = "raw") # use actual predictions

#Probabilities
pred_glm_prob <- predict.train(model_glm,
                               newdata = testing,
                               type = "prob") # use the probabilities

#Confusion matrices
confusionMatrix(data = pred_glm_raw,
                factor(testing$y),
                positive = "yes")

#Calculate ROC
rocr.pred.lr = prediction(predictions = as.numeric(pred_glm_raw), labels = as.numeric(testing$y))
rocr.perf.lr = performance(rocr.pred.lr, measure = "tpr", x.measure = "fpr")
rocr.auc.lr = as.numeric(performance(rocr.pred.lr, "auc")@y.values)

#Print ROC AUC
rocr.auc.lr

#Plot ROC curve
plot(rocr.perf.lr,
     lwd = 3, colorize = TRUE,
     print.cutoffs.at = seq(0, 1, by = 0.1),
     text.adj = c(-0.2, 1.7),
     main = 'ROC Curve')
mtext(paste('Logistic Regression - auc : ', round(rocr.auc.lr, 5)))

abline(0, 1, col = "red", lty = 2)

#Explain models
p_fun <- function(object, newdata) {
  predict(object,
          newdata = newdata,
          type = "prob")[,2]
}
yTest <- ifelse(testing$y == "yes", 1, 0)
# create an explainer
explainer_classif_glm <- DALEX::explain(model_glm, label = "glm",
                                        data = testing, y = yTest,
                                        predict_function = p_fun)

#Calculate model predictions and residuals
mp_classif_glm <- model_performance(explainer_classif_glm)
mp_classif_glm
#Plot performance
plot(mp_classif_glm)

#End of Code
