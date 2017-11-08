#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "cross validation"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday November 8, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' 
#' 
#' Outline
#' ===
#' 
#' - cross validation to evaluate machine learning accuracy
#'     - holdout method
#'     - k-fold cross validation
#'     - leave one out cross validation
#' - cross validation to select tuning parameters
#'     - lasso
#'     - KNN
#' - Combine tuning parameter selection and evaluate prediction accuracy
#' - A common mistake about machine learning feature selection
#' 
#' 
#' 
#' Cross validation to evaluate machine learning accuracy
#' ===
#' 
#' ![](../figure/crossValidation.png)
#' 
#' 
#' holdout method
#' ===
#' 
#' - Simplest kind of cross validation.
#' - The data set is split into training set and the testing set.
#'     - Model the machine learning classifier using training set only.
#'     - Evaluate the prediction error only using the testing set.
#' 
#' - Pros: easy to implement
#' - Cons: its evaluation can have a high variance.
#'     - depend heavily on which data points end up in the training set and which end up in the test set
#' 
#' 
#' holdout method example
#' ===
#' 
## ------------------------------------------------------------------------
library(ElemStatLearn)
library(randomForest)

prostate0 <- prostate
prostate0$svi <- as.factor(prostate0$svi)
trainIabel <- prostate$train
prostate0$train <- NULL

prostate_train <- subset(prostate0, trainIabel)
prostate_test <- subset(prostate0, !trainIabel)

rfFit <- randomForest(svi ~ ., data=prostate_train)
rfFit

rfPred <- predict(rfFit, prostate_test)
rfTable <- table(rfPred, truth=prostate_test$svi)
rfTable
1 - sum(diag(rfTable))/sum(rfTable)

#' 
#' 
#' k-fold cross validation
#' ===
#' 
#' - Only split the data into two parts may result in high variance.
#' - Another commonly used approach is to split the data into $K$ folds.
#' - Normally $K = 5$ or $K = 10$ are recommended to balance the bias and variance.
#' 
#' ![](../figure/k-fold.png)
#' 
#' 
#' Algorithm:
#' 
#' - Split the original dataset $D$ into $K$ folds, and obtain $D_1, \ldots, D_k,\ldots, D_K$ such that $D_1, \ldots, D_K$ are disjoint; size of $D_k$ $|D_k|$ is roughly the same for different $k$ and $\sum_k |D_k| = |D|$.
#' 
#' - For each iteration $k = 1, \ldots, K$,
#' use the $D_k$ as the testing dataset and $D_{-k} = D - D_k$ as the training dataset. Build the classifier $f^k$ based on $D_{-k}$. Then predict $D_k$ and obtain the prediction error rate $E_k$
#' 
#' - Use $E = \frac{1}{K} \sum_k E_k$ as the final prediction error rate.
#' 
#' 
#' k-fold cross validation example
#' ===
#' 
## ------------------------------------------------------------------------
#Create K = 5 equally size folds

K <- 5
folds <- cut(seq(1,nrow(prostate0)),breaks=5,labels=FALSE)

EKs <- numeric(K)

for(k in 1:K){
  atrain <- subset(prostate0, folds != k)
  atest <- subset(prostate0, folds == k)

  krfFit <- randomForest(svi ~ ., data=atrain)
  krfPred <- predict(krfFit, atest)
  krfTable <- table(krfPred, truth=atest$svi)
  
  EKs[k] <- 1 - sum(diag(krfTable))/sum(krfTable)
}

EKs

mean(EKs)

#' 
#' Leave one out cross validation
#' ===
#' 
#' For K-fold cross validation
#' 
#' - $K = 2$, holdout method
#' - $K = n$, leave one out cross validation
#'     - Each time, only use one sample as the testing sample and the rest of all sample as the training data.
#'     - Iterate total $n$ times.
#' 
#' 
#' 
#' leave one out cross validation example
#' ===
#' 
## ------------------------------------------------------------------------
n <- nrow(prostate0)

EKs <- numeric(n)

for(k in 1:n){
  #cat("leaving the",k,"sample out!","\n")
  atrain <- prostate0[-k,]
  atest <- prostate0[k,]

  krfFit <- randomForest(svi ~ ., data=atrain)
  krfPred <- predict(krfFit, atest)
  krfTable <- table(krfPred, truth=atest$svi)
  
  EKs[k] <- 1 - sum(diag(krfTable))/sum(krfTable)
}

EKs

mean(EKs)

#' 
#' Important note for cross validation
#' ===
#' 
#' - the classifier should only build on the training data.
#' - the classifier should not use the information of the testing data.
#' 
#' 
#' 
#' Choosing a value of the tuning parameter
#' ===
#' 
#' Some machine learning classifiers contain tuning parameters
#' 
#' - CART: pruning criteria.
#' - K-nearest-neighbors (KNN): K (will be introduced shortly)
#' - Lasso: regularization penalty $\lambda$.
#' 
#' 
#' The tuning parameter is crucial since it controls the amount of regularization (model complexity).
#' 
#' - We also refer selecting different tuning parameter values (corresponding to different fitted model) as **Model selection**.
#' 
#' 
#' Model selection
#' ===
#' 
#' 
#' ![](../figure/overfitting.jpg)
#' 
#' 
#' Lasso problem
#' ===
#' 
#' $$\hat{\beta}^{lasso} = \arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$
#' 
## ------------------------------------------------------------------------
library(lars)
library(ggplot2)
options(stringsAsFactors=F)
x <- as.matrix(prostate[,1:8])
y <- prostate[,9]
lassoFit <- lars(x, y) ## lar for least angle regression

lambdas <- seq(0,5,0.5)
coefLassoFit <- coef(lassoFit, s=lambdas, mode="lambda") 
adataFrame <- NULL
for(i in 1:ncol(coefLassoFit)){
  acoef <- colnames(coefLassoFit)[i]
  adataFrame0 <- data.frame(lambda=lambdas, coef = coefLassoFit[,i], variable=acoef)
  adataFrame <- rbind(adataFrame, adataFrame0)
}

ggplot(adataFrame) + aes(x=lambda, y=coef, col=variable) + 
  geom_point() +
  geom_path()

#' 
#' 
#' How to select the best tuning parameter
#' ===
#' 
#' - Use cross validation. 
#' - We should choose the tuning parameter such that the cross validation error is minimized.
#' 
#' ![](../figure/CVerror.png)
#' 
#' Tuning parameter selection algorithm
#' ===
#' 
#' - Split the original dataset $D$ into $K$ folds, and obtain $D_1, \ldots, D_k,\ldots, D_K$ such that $D_1, \ldots, D_K$ are disjoint; size of $D_k$ $|D_k|$ is roughly the same for different $k$ and $\sum_k |D_k| = |D|$.
#' 
#' - pre-specify a range of tuning parameters $\lambda_1, \ldots, \lambda_B$
#' 
#' - For each iteration $k = 1, \ldots, K$,
#' use the $D_k$ as the testing dataset and $D_{-k} = D - D_k$ as the training dataset. Build the classifier $f^k_{\lambda_b}$ based on $D_{-k}$ and tuning parameter $\lambda_b$, $(1 \le b \le B)$. Then predict $D_k$ and obtain the 
#'     - mean squared error (MSE) or root mean squared error (rMSE) $e_k(\lambda_b)$ for continuous outcome.
#'     - classification error rate for categorical outcome variable. 
#' 
#' - Use $e(\lambda_b) = \frac{1}{K} \sum_k e_k(\lambda_b)$ as the average (MSE, rMSE or classification error rate) for tuning parameter $(\lambda_b)$.
#' 
#' - Choose the tuning parameter $\hat{\lambda} = \arg \min_{\lambda\in (\lambda_1, \ldots, \lambda_B)} e(\lambda)$
#' 
#' $$e(\lambda) = \frac{1}{K} \sum_{k=1}^K \sum_{i \in D_k} (y_i - \hat{f}_\lambda^{-k}(x_i))$$
#' 
#' Implement tuning parameter selection for lasso problem
#' ===
#' 
## ------------------------------------------------------------------------
K <- 10
folds <- cut(seq(1,nrow(prostate)),breaks=K,labels=FALSE)
lambdas <- seq(0.5,5,0.5)

MSEs <- matrix(NA,nrow=length(lambdas),ncol=K)
rownames(MSEs) <- lambdas
colnames(MSEs) <- 1:K

for(k in 1:K){
  atrainx <- as.matrix(prostate[!folds==k,1:8])
  atestx <- as.matrix(prostate[folds==k,1:8])
 
  atrainy <- prostate[!folds==k,9]
  atesty <- prostate[folds==k,9]

  klassoFit <- lars(atrainx, atrainy) ## lar for least angle regression

  predictValue <- predict.lars(klassoFit, atestx, s=lambdas, type="fit",  mode="lambda")
  MSE <- apply(predictValue$fit,2, function(x) sqrt(sum((x - atesty)^2)))
  
  MSEs[,k] <- MSE
}

MSEdataFrame <- data.frame(lambda=lambdas, MSE = rowMeans(MSEs))
ggplot(MSEdataFrame) + aes(x=lambda, y = MSE) + geom_point() + geom_path()

#' 
#' lars example
#' ===
#' 
#' Refer to cv.lars() help file example
#' 
#' KNN motivating example
#' ===
#' 
#' 
## ------------------------------------------------------------------------
library(MASS)
set.seed(32611)
N<-100
d1<-mvrnorm(N, c(1,-1), matrix(c(2, 1, 1, 2), 2, 2))
d2<-mvrnorm(N, c(-1,1), matrix(c(2, 1, 1, 2), 2, 2))
atest <- rbind(c(1,1),c(2,-2),c(-2,2))
d <- rbind(d1, d2,atest)
label <- c(rep("A", N), rep("B", N),rep("test",3))
text <- label
text[label!="test"] <- ""
data_train <- data.frame(label=label, x=d[,1], y=d[,2], text=text)
names(data_train)<-c("label", "x", "y", "text")

ggplot(data_train, aes(x=x,y=y,col=label, label = text)) + 
  geom_point() + geom_text(vjust = -1,size=5)

#' 
#' 
#' KNN algorithm
#' ===
#' 
#' - pre-specify $K$ (e.g. $K = 5$).
#' - for each testing sample, find the top $K$ training samples that are closest to the testing sample.
#' - The label of the testing sample will depend on the majority vote of the training labels for categorical outcome.
#' 
#' $$\hat{f}(x) = mode_{i \in N_k(x)} y_i$$
#' 
#' 
#' - KNN regression
#' 
#' $$\hat{f}(x) = \frac{1}{K} \sum_{i \in N_k(x)} y_i$$
#' 
#' - KNN missing value imputation
#' 
#' 
#' KNN example
#' ===
#' 
## ------------------------------------------------------------------------
library(ElemStatLearn)
library(class)

prostate0 <- prostate
prostate0$svi <- as.factor(prostate0$svi)
trainIabel <- prostate$train

adata_train <- prostate0[trainIabel, -5]
adata_test <- prostate0[!trainIabel, -5]

alabel_train <- prostate0[trainIabel, 5]
alabel_test <- prostate0[!trainIabel, 5]

knnFit <- knn(adata_train, adata_test, alabel_train, k=5)

knnTable <- table(knnFit, truth=alabel_test)
knnTable
1 - sum(diag(knnTable))/sum(knnTable)

#' 
#' 
#' KNN Exercise
#' ===
#' 
#' - visualize the decision boundary for KNN (with KNN motivating example).
#' - use cross validation to determine the optimum $K$ for KNN (with prostate cancer data).
#' 
#' 
#' Combine selecting tuning parameters and evaluating prediction accuracy
#' ===
#' 
#' 1. Split the data into training set and testing set.
#' 2. In training set, use cross validation to determine the best tuning parameter.
#' 3. Use the best tuning parameter and the entire training set to build a classifier.
#' 4. Evaluation:
#'     - Report the final prediction model
#'     - Predict the testing set
#'     - Report the accuracy.
#' 
#' For **Step 1**, can we also use cross validation?
#' 
#' Common mistake about tuning parameter selection in machine learning
#' ===
#' 
#' ![](../figure/tuning.png)
#' 
#' 
#' Nested Cross validation
#' ===
#' 
#' 1. Use K-folder cross validation (outer) to split the original data into training set and testing set.
#' 2. For $k^{th}$ fold training set, use cross validation (inner) to determine the best tuning parameter of the $k^{th}$ fold.
#' 3. Use the best tuning parameter of the $k^{th}$ fold and the $k^{th}$ fold training set to build a classifier.
#' 4. Predict the $k^{th}$ fold testing set and report the accuracy.
#' 5. Evaluation:
#'     - Report the average accuracy from outer K-folder cross validation..
#' 
#' 
#' 
#' Feature selection in machine learning
#' ===
#' 
#' A small subset of breast cancer gene expression data (BRCA) comparing tumor subjects with normal subjects.
#' The data is in file brca_tcga.csv ([Download here](https://caleb-huo.github.io/teaching/2017FALL/HW/HW1/brca_tcga.csv)), with each row representing a gene and each column representing a subject.
#' 
#' - 200 genes
#' - 40 samples
#' 
#' If we directly use all these features to build a classifier,
#' we may include too many noises.
#' The model is also too complicated.
#' 
#' Feature selection: only keep top genes most associated with the class labels
#' (e.g. 20 genes and 40 samples)
#' 
#' - Criteria: t-test p-value
#' 
#' Feature selection in machine learning
#' ===
#' 
#' ![](../figure/filtering.png)
#' 
#' 
#' Feature selection should also be thought as tuning parameter selection and model selection.
#' 
#' 
#' Resources
#' ===
#' 
#' - <https://www.openml.org/a/estimation-procedures/1>
#' - <https://www.cs.cmu.edu/~schneide/tut5/node42.html>
#' - <http://www.stat.cmu.edu/~ryantibs/datamining/lectures/18-val1.pdf>
#' - <http://www.stat.cmu.edu/~ryantibs/datamining/lectures/19-val2.pdf>
#' - Element of Statistical learning, Chapter 7
#' 
## ------------------------------------------------------------------------
knitr::purl("crossValidation.rmd", output = "crossValidation.R ", documentation = 2)

