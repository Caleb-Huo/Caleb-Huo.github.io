#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "cross validation"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday November 23, 2020"
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
#' - Other machine learning approaches
#'     - K nearest neighbour (KNN)
#'     - Linear discriminant analysis (LDA)
#'     - Support vector machine (SVM) 
#' 
#' 
#' 
#' 
#' Holdout method to evaluate machine learning accuracy
#' ===
#' 
#' ![](../figure2/crossValidation.png)
#' 
#' 
#' - Pros: easy to implement
#' - Cons: its evaluation may have a high variance.
#'     - depend heavily on which data points in the training set and which in the test set
#' 
#' 
#' Holdout method example
#' ===
#' 
## ------------------------------------------------------------------------
library(ElemStatLearn)
library(randomForest)

prostate0 <- prostate
prostate0$svi <- as.factor(prostate0$svi)
trainLabel <- prostate$train
prostate0$train <- NULL

prostate_train <- subset(prostate0, trainLabel)
prostate_test <- subset(prostate0, !trainLabel)

rfFit <- randomForest(svi ~ ., data=prostate_train)
rfFit

rfPred <- predict(rfFit, prostate_test)
rfTable <- table(rfPred, truth=prostate_test$svi)
rfTable
1 - sum(diag(rfTable))/sum(rfTable) ## error rate

#' 
#' 
#' k-fold cross validation
#' ===
#' 
#' - Only split the data into two parts may result in high variance.
#' - Another commonly used approach is to split the data into $K$ folds.
#' - Normally $K = 5$ or $K = 10$ are recommended to balance the bias and variance.
#' 
#' ![](../figure2/k-fold.png)
#' 
#' 
#' Algorithm:
#' 
#' - Split the original dataset $D$ into $K$ folds, and obtain $D_1, \ldots, D_k,\ldots, D_K$ such that $D_1, \ldots, D_K$ are disjoint; $|D_k|$ (size of $D_k$) is roughly the same for different $k$ and $\sum_k |D_k| = |D|$.
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
#' Choosing tuning parameters
#' ===
#' 
#' Some machine learning classifiers contain tuning parameters
#' 
#' - CART: pruning criteria.
#' - K-nearest-neighbors (KNN): K (will be introduced shortly)
#' - Lasso: regularization penalty $\lambda$.
#' 
#' The tuning parameter is crucial since it controls the amount of regularization (model complexity).
#' 
#' 
#' 
#' Tuning parameter selection
#' ===
#' 
#' 
#' ![](../figure2/overfitting.jpg)
#' 
#' 
#' Lasso problem
#' ===
#' 
#' $$\hat{\beta}^{lasso} = \frac{1}{2}\arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$
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
#' ![](../figure2/CVerror.png)
#' 
#' Tuning parameter selection algorithm
#' ===
#' 
#' - Split the original dataset $D$ into $K$ folds, and obtain $D_1, \ldots, D_k,\ldots, D_K$ such that $D_1, \ldots, D_K$ are disjoint; $|D_k|$ (size of $D_k$) is roughly the same for different $k$ and $\sum_k |D_k| = |D|$.
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
#' ---
#' 
#' Choose the tuning parameter $\hat{\lambda} = \arg \min_{\lambda\in (\lambda_1, \ldots, \lambda_B)} e(\lambda)$
#' 
#' - Continuous outcome MSE: $$e(\lambda) = \frac{1}{K} \sum_{k=1}^K \sum_{i \in D_k} (y_i - \hat{f}_\lambda^{-k}(x_i))^2$$
#' 
#' - Continuous outcome rMSE: $$e(\lambda) = \frac{1}{K} \sum_{k=1}^K \sqrt{\sum_{i \in D_k} (y_i - \hat{f}_\lambda^{-k}(x_i))^2}$$
#' 
#' - Categorical outcome variable classification error rate: $$e(\lambda) = \frac{1}{K} \sum_{k=1}^K \frac{1}{|D_k|}\sum_{i \in D_k} \mathbb{I}(y_i \ne \hat{f}_\lambda^{-k}(x_i))$$
#' 
#' 
#' Implement tuning parameter selection for lasso problem
#' ===
#' 
## ------------------------------------------------------------------------
K <- 10
folds <- cut(seq(1,nrow(prostate)),breaks=K,labels=FALSE)
lambdas <- seq(0.5,5,0.5)

rMSEs <- matrix(NA,nrow=length(lambdas),ncol=K)
rownames(rMSEs) <- lambdas
colnames(rMSEs) <- 1:K

for(k in 1:K){
  atrainx <- as.matrix(prostate[!folds==k,1:8])
  atestx <- as.matrix(prostate[folds==k,1:8])
 
  atrainy <- prostate[!folds==k,9]
  atesty <- prostate[folds==k,9]

  klassoFit <- lars(atrainx, atrainy) ## lar for least angle regression

  predictValue <- predict.lars(klassoFit, atestx, s=lambdas, type="fit",  mode="lambda")
  rMSE <- apply(predictValue$fit,2, function(x) sqrt(sum((x - atesty)^2))) ## square root MSE
  
  rMSEs[,k] <- rMSE
}

rMSEdataFrame <- data.frame(lambda=lambdas, rMSE = rowMeans(rMSEs))
ggplot(rMSEdataFrame) + aes(x=lambda, y = rMSE) + geom_point() + geom_path()

#' 
#' Common mistake about tuning parameter selection in machine learning
#' ===
#' 
#' ![](../figure2/tuning.png)
#' 
#' 
#' Combine selecting tuning parameters and evaluating prediction accuracy 
#' ===
#' 
#' 1. Split the data into training set and testing set.
#' 2. In training set, use cross validation to determine the best tuning parameter.
#' 3. Use the best tuning parameter and the entire training set to build a classifier.
#' 4. Evaluation:
#'     - Predict the testing set
#'     - Report the accuracy.
#' 
#' For **Step 1**, can we also use cross validation?
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
#' Other machine learning approaches
#' ===
#' 
#' - K nearest neighbour (KNN)
#' - Linear discriminant analysis (LDA)
#' - Support vector machine (SVM) 
#' 
#' 
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
#' - pre-specify $K$ (e.g., $K = 5$).
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
trainLabel <- prostate$train

## here 5 is the index of svi in the prostate data
adata_train <- prostate0[trainLabel, -5]
adata_test <- prostate0[!trainLabel, -5]

alabel_train <- prostate0[trainLabel, 5]
alabel_test <- prostate0[!trainLabel, 5]

knnFit <- knn(adata_train, adata_test, alabel_train, k=5)

knnTable <- table(knnFit, truth=alabel_test)
knnTable
1 - sum(diag(knnTable))/sum(knnTable)

#' 
#' 
#' KNN Exercise (Will be in HW)
#' ===
#' 
#' - use cross validation to determine the optimum $K$ for KNN (with prostate cancer data).
#' 
#' Linear discriminant analysis (LDA)
#' ===
#' 
#' Problem setting:
#' 
#' - $X \in \mathbb{R}^p$ is a sample with $p$ features.
#' - $Y$ is the class label, with possible value $1$ and $2$.
#' - Suppose we have training dataset:
#'     - $X_{train} \in \mathbb{R}^{n \times p}$
#'     - $Y_{train} \in \{1,2\}^n$
#' - Goal:
#'     - build a classifier
#'     - predict testing data $X_{test}$
#'     
#' - Gaussian Assumption:
#' 
#' $$f(X|Y=k) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp ( - \frac{1}{2} (X - \mu_k)^\top \Sigma_k^{-1} (X - \mu_k))$$
#' 
#' Intuition for Linear discriminant analysis (LDA)
#' ===
#' 
## ---- cache=T------------------------------------------------------------
library(MASS)
library(ggplot2)

set.seed(32611)
N<-100
d1<-mvrnorm(N, c(1,-1), matrix(c(2, 1, 1, 2), 2, 2))
d2<-mvrnorm(N, c(-1,1), matrix(c(2, 1, 1, 2), 2, 2))
d <- rbind(d1, d2)
label <- c(rep("1", N), rep("2", N))
data_train <- data.frame(label=label, x1=d[,1], x2=d[,2])
names(data_train)<-c("label", "x1", "x2")

ggplot(data_train) + aes(x=x1,y=x2,col=label) + geom_point() +  stat_ellipse()

#' 
#' 
#' - Intuition:
#'     - We estimate the Gaussian density function for each class, and for a future testing sample, the classifier is "select the class with larger density function value".
#' 
#' 
#' 
#' Math behind Linear discriminant analysis (LDA)
#' ===
#' 
#' - Gaussian density:
#' $$f(X|Y=k) = \frac{1}{(2\pi)^{p/2} |\Sigma_k|^{1/2}} \exp ( - \frac{1}{2} (X - \mu_k)^\top \Sigma_k^{-1} (X - \mu_k))$$
#' 
#' - The Bayes rule:
#' 
#' $\begin{aligned}
#' C_{Bayes}(X) & = \arg \max_k f(Y=k | X) \\
#' & = \arg \max_k \frac{f(Y=k)f(X|Y=k)}{f(X)} \\
#' & = \arg \max_k f(Y=k)f(X|Y=k) \\
#' & = \arg \max_k \log f(Y=k) + \log f(X|Y=k) \\
#' & = \arg \max_k \log f(Y=k) - \frac{1}{2} \log |\Sigma_k| -  \frac{1}{2} (X - \mu_k)^\top \Sigma_k^{-1} (X - \mu_k) \\
#' & = \arg \min_k - 2 \log f(Y=k) + \log |\Sigma_k| + (X - \mu_k)^\top \Sigma_k^{-1} (X - \mu_k)
#' \end{aligned}$
#' 
#' 
#' 
#' Decision boundary for Linear discriminant analysis (LDA)
#' ===
#' 
#' The decision boundary for class $m$ and class $l$ is:
#' $$f(Y=l|X) = f(Y=m|X)$$
#' 
#' Or equivalently,
#' 
#' $\begin{aligned}
#' & - 2 \log f(Y=l) + \log |\Sigma_l| + (X - \mu_l)^\top \Sigma_l^{-1} (X - \mu_l) \\
#' & = - 2 \log f(Y=m) + \log |\Sigma_m| + (X - \mu_m)^\top \Sigma_m^{-1} (X - \mu_m)
#' \end{aligned}$
#' 
#' Further assume uniform prior $f(Y=l) = f(Y=m)$, and same covariance structure $\Sigma_l = \Sigma_m = \Sigma$,
#' the decision hyperlane simplifies as:
#' 
#' $$2 (\mu_l - \mu_m)^\top \Sigma^{-1} X - (\mu_l - \mu_m)^\top \Sigma^{-1} (\mu_l + \mu_m) = 0$$
#' 
#' $$2 (\mu_l - \mu_m)^\top \Sigma^{-1} (X - \frac{\mu_l + \mu_m}{2}) = 0$$
#' 
#' 
#' You can see the decision boundary is linear and passes the center between $\frac{\mu_l + \mu_m}{2}$.
#' 
#' 
#' 
#' Visualize the decision boundary (LDA)
#' ===
#' 
## ---- cache=T------------------------------------------------------------
library(MASS)

alda <- lda(label ~ . ,data=data_train)

avec <- seq(-4,4,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x1=z[,1],x2=z[,2])
predict_lda <- predict(alda, z)$class

z_lda <- cbind(region=as.factor(predict_lda), z)
ggplot(z_lda) + aes(x=x1,y=x2,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x1,y=x2,col=label)) +
  labs(title = "LDA boundary")

#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' Quadratic discriminant analysis (QDA)
#' ===
#' 
#' Further assume uniform prior $f(Y=l) = f(Y=m)$, but different covariance structure $\Sigma_l$, $\Sigma_m$,
#' the decision hyperlane simplifies as:
#' 
#' $\begin{aligned}
#'  \log |\Sigma_l| + (X - \mu_l)^\top \Sigma_l^{-1} (X - \mu_l) =  \log |\Sigma_m| + (X - \mu_m)^\top \Sigma_m^{-1} (X - \mu_m)
#' \end{aligned}$
#' 
#' 
#' You will see the decision boundary is a quadratic function.
#' 
#' 
## ---- cache=T------------------------------------------------------------
library(MASS)
library(ggplot2)

set.seed(32611)
N<-100
d1<-mvrnorm(N, c(0,2), matrix(c(4, 0, 0, 1), 2, 2))
d2<-mvrnorm(N, c(0,-2), matrix(c(1, 0, 0, 4), 2, 2))
d <- rbind(d1, d2)
label <- c(rep("1", N), rep("2", N))
data_train <- data.frame(label=label, x1=d[,1], x2=d[,2])
names(data_train)<-c("label", "x1", "x2")

ggplot(data_train) + aes(x=x1,y=x2,col=label) + geom_point() +  stat_ellipse()

#' 
#' 
#' Visualize the decision boundary (QDA)
#' ===
#' 
## ---- cache=T------------------------------------------------------------
library(MASS)
aqda <- qda(label ~ . ,data=data_train)

avec <- seq(-6,6,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x1=z[,1],x2=z[,2])
predict_qda <- predict(aqda, z)$class

z_qda <- cbind(region=as.factor(predict_qda), z)
ggplot(z_qda) + aes(x=x1,y=x2,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x1,y=x2,col=label)) +
  labs(title = "QDA boundary")

#' 
#' 
#' Support vector machine (SVM)
#' ===
#' 
#' - Motivation: find a boundary to separate data from with different labels
#' 
#' - Linear separable case
#' 
## ------------------------------------------------------------------------
library(MASS)
library(ggplot2)

set.seed(32611)
N<-100
d1<-mvrnorm(N, c(3,-3), matrix(c(1, 0, 0, 1), 2, 2))
d2<-mvrnorm(N, c(-3,3), matrix(c(1, 0, 0, 1), 2, 2))
d <- rbind(d1, d2)
label <- c(rep("1", N), rep("-1", N))
data_train <- data.frame(label=label, x1=d[,1], x2=d[,2])
names(data_train)<-c("label", "x1", "x2")

p <- ggplot(data_train) + aes(x=x1,y=x2,col=label) + geom_point()
p + geom_abline(slope=1,intercept=0, col="black") + 
 geom_abline(slope=0.8,intercept=0.5, col="green") + 
 geom_abline(slope=1.4,intercept=-0.4, col = "orange")

#' 
#' SVM margin
#' ===
#' 
## ------------------------------------------------------------------------
p +
geom_abline(slope=1,intercept=max(d1[,2] - d1[,1]), col="black", linetype = "dashed") +
geom_abline(slope=1,intercept=min(d2[,2] - d2[,1]), col="black", linetype = "dashed") +
geom_abline(slope=1,intercept=(max(d1[,2] - d1[,1]) + min(d2[,2] - d2[,1]))/2, col="black")

#' 
#' - Black line (Separation boundary) with intercept included:
#' $$f(x) = w^\top x = w_1 x_1 + w_2 x_2 + c$$
#' 
#' - red dots
#' $$w^\top x \le -1$$
#' Otherwise, we can re-scale $w$ to make sure this relationship holds
#' 
#' - blue dots
#' $$w^\top x \ge 1$$
#' 
#' - Unify red dots and blue dots:
#' $$y w^\top x \ge 1$$
#' 
#' 
#' Formulate optimization problem via SVM margin
#' ===
#' 
#' - Upper dashed line:
#' $$w^\top x + 1 = 0$$
#' - lower dashed line:
#' $$w^\top x- 1 = 0$$
#' 
#' - Margin:
#' 
#' $$\frac{w^\top x + 1 - (w^\top x - 1)}{2\|w\|_2} = \frac{2}{2\|w\|_2} = \frac{1}{\|w\|_2}$$
#' 
#' - Optimization problem:
#' 
#' $\max_{w} \frac{1}{\|w\|_2}$ subject to $y_i (w^\top x_i ) \ge 1$
#' 
#' 
#' 
#' SVM: large Margin Classifier
#' ===
#' 
#' - $\frac{1}{\|w\|_2}$ is not easy to maximize, but $\|w\|_2^2$ is easy to minimize: 
#' 
#' - SVM (linear separable case):
#' 
#' $\min_{w} \frac{1}{2} \|w\|_2^2$ subject to $y_i (w^\top x_i ) \ge 1$
#' This is the SVM primal objective function.
#' 
#' - Lagrange function:
#' 
#' $$L(w, \alpha) = \frac{1}{2} \|w\|_2^2 - \sum_i \alpha_i [y_i (w^\top x_i ) - 1]$$
#' By KKT condition, $\frac{\partial L}{\partial w}$ need to be 0 at the optimun solution.
#' 
#' $$\frac{\partial L}{\partial w_j} = w_j - \sum_i \alpha_i y_i x_{ij} = 0$$
#' $$w_j = \sum_i \alpha_i y_i x_{ij}$$
#' 
#' Plugging terms back into Lagrange function $L$:
#' $$\min_\alpha  \frac{1}{2} \sum_{i,i'} \alpha_i \alpha_{i'} y_i y_{i'} x_i^\top x_{i'}$$
#' Subject to $\alpha_i \ge 0$
#' 
#' This is the SVM dual objective function.
#' 
#' Geometric interpretation of SVM
#' ===
#' 
## ------------------------------------------------------------------------
p +
geom_abline(slope=1,intercept=max(d1[,2] - d1[,1]), col="black", linetype = "dashed") +
geom_abline(slope=1,intercept=min(d2[,2] - d2[,1]), col="black", linetype = "dashed") +
geom_abline(slope=1,intercept=(max(d1[,2] - d1[,1]) + min(d2[,2] - d2[,1]))/2, col="black")

#' 
#' From KKT condition:
#' 
#' $$\alpha_i [y_i w^\top x_i - 1] = 0$$
#' 
#' - When $\alpha_i = 0$, no restriction on $y_i w^\top x_i - 1$, data $i$ is beyond the boundary.
#' - When $\alpha_i > 0$, $y_i w^\top x_i = 1$, data $i$ is on the boundary, which are called support vectors.
#' 
#' 
#' Use svm function in R
#' ===
#' 
## ------------------------------------------------------------------------
library(e1071)
library(ggplot2)

set.seed(32611)
N<-100
d1<-mvrnorm(N, c(3,-3), matrix(c(1, 0, 0, 1), 2, 2))
d2<-mvrnorm(N, c(-3,3), matrix(c(1, 0, 0, 1), 2, 2))
d <- rbind(d1, d2)
label <- c(rep("1", N), rep("-1", N))
data_train <- data.frame(label=factor(label), x1=d[,1], x2=d[,2])
names(data_train)<-c("label", "x1", "x2")

asvm <- svm(label ~ x1 + x2 ,data=data_train, kernel = "linear")

avec <- seq(-6,6,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x1=z[,1],x2=z[,2])
predict_svm <- predict(asvm, z)

z_svm <- cbind(region=as.factor(predict_svm), z)
ggplot(z_svm) + aes(x=x1,y=x2,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x1,y=x2,col=label)) +
  labs(title = "SVM boundary")

#' 
#' 
#' 
#' 
#' 
#' Property of SVM
#' ===
#' 
#' $$w_j = \sum_i \alpha_i y_i x_{ij}$$
#' 
#' - Weight vector $w$ is weighted linear combination of samples.
#' - Since $\alpha_i \ne 0$ only if the samples are on the margin, so only the margin samples contribute to $w$.
#' - This is quadratic programming so optimization is not too hard.
#' 
#' $$\min_\alpha  \frac{1}{2} \sum_{i,i'} \alpha_i \alpha_{i'} y_i y_{i'} x_i^\top x_{i'} $$
#' Subject to $\alpha_i \ge 0$
#' 
#' - In the dual formulation, $x$ plays a part only through $x_i^\top x_j$, which can be further extended as kernel:
#' $k(x_i, x_j)$
#'     - Linear kernel: $k(x_i, x_j) = x_i^\top x_j$
#'     - Gaussian kernel: $k(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|_2^2)$
#'     - polynomial kernel 
#'     - sigmoid kernel
#' 
#' 
#' Use svm function with Gaussian kernel in R 
#' ===
#' 
## ------------------------------------------------------------------------
library(e1071)
library(ggplot2)

N<-100
set.seed(32611)
theta1 <- runif(N,0,2*pi)
r1 <- runif(N,2,3)

theta2 <- runif(N,0,2*pi)
r2 <- runif(N,4,5)

d1 <- cbind(r1 * cos(theta1), r1 * sin(theta1))
d2 <- cbind(r2 * cos(theta2), r2 * sin(theta2))

d <- rbind(d1, d2)
label <- c(rep("1", N), rep("-1", N))
data_train <- data.frame(label=factor(label), x=d[,1], y=d[,2])
names(data_train)<-c("label", "x", "y")

ggplot(data_train, aes(x=x,y=y,col=label))+ geom_point() 

#' 
#' Use svm function with Gaussian kernel in R 
#' ===
#' 
## ------------------------------------------------------------------------
asvm <- svm(label ~ . ,data=data_train, kernel = "radial")

avec <- seq(-6,6,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x=z[,1],y=z[,2])
predict_svm <- predict(asvm, z)

z_svm <- cbind(region=as.factor(predict_svm), z)
ggplot(z_svm) + aes(x=x,y=y,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x,y=y,col=label)) +
  labs(title = "SVM boundary")

#' 
#' Interpretation for Kernel SVM:
#' ===
#' 
#' - The data is not linearly separable in two dimensional space
#' - Kernel method can create new features, thus map the two dimensional space to higher dimensional space
#' 
#' $$\exp(x) = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \ldots$$
#' 
#' - The data are linearly separable in this higher dimensional space
#' 
#' ![](../figure2/svmHigh.png)
#' 
#' 
#' 
#' SVM when linear separator is impossible
#' ===
#' 
## ------------------------------------------------------------------------
library(MASS)
library(ggplot2)

set.seed(32611)
N<-100
d1<-mvrnorm(N, c(2,-2), matrix(c(2, 1, 1, 2), 2, 2))
d2<-mvrnorm(N, c(-2,2), matrix(c(2, 1, 1, 2), 2, 2))
d <- rbind(d1, d2)
label <- c(rep("1", N), rep("-1", N))
data_train <- data.frame(label=factor(label), x=d[,1], y=d[,2])
names(data_train)<-c("label", "x", "y")

p <- ggplot(data_train) + aes(x=x,y=y,col=label) + geom_point()
p + geom_abline(slope=1,intercept=(max(d1[,2] - d1[,1]) + min(d2[,2] - d2[,1]))/2, col="black")



#' 
#' SVM for linearly no-separable case
#' ===
#' 
#' $$\min_{w} \frac{1}{2} \|w\|_2^2 + C\|\zeta\|_2^2$$ 
#' subject to $y_i (w^\top x_i ) \ge 1 - \zeta_i$ and  $\zeta_i \ge 0$
#' 
#' - Similar we can write down the Lagrange function and use KKT condition to get the dual problem.
#' 
#' 
#' Use svm function in R with non-linear separable case
#' ===
#' 
## ------------------------------------------------------------------------
library(e1071)
library(ggplot2)

asvm <- svm(label ~ . ,data=data_train, kernel = "linear")

avec <- seq(-6,6,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x=z[,1],y=z[,2])
predict_svm <- predict(asvm, z)

z_svm <- cbind(region=as.factor(predict_svm), z)
ggplot(z_svm) + aes(x=x,y=y,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x,y=y,col=label)) +
  labs(title = "SVM boundary")

#' 
#' Summary for classification functions in R
#' ===
#' 
#' Classifier  | package | function
#' ------------- | ------------- | -------------
#' Logistic regression   | stats | glm with parameter family=binomial()
#' Linear and quadratic discriminant analysis      | MASS | lda, qda
#' DLDA and DQDA      | sma | stat.diag.da
#' KNN classification      | class | knn
#' CART      | rpart | rpart
#' Random forest      | randomForest | randomForest
#' Support Vector machines      | e1071 | svm
#' 
#' 
#' Deep learning
#' ===
#' 
#' - Have better accuracy compared to other exisiting machine learning method
#' - Free online deep learning course
#'   - https://cds.nyu.edu/deep-learning/
#'   - Instructor: Yann LeCun
#'     - 2028 Tuning Awardee for his contribution in deep learning
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
