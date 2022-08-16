#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday November 15, 2021"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Supervised Machine Learning
#' ---
#' 
#' 
#' Outline
#' ===
#' 
#' - Supervised machine learning
#'   - Classification and regression tree
#'   - Random forest
#'   - Others (on Wednesday)
#' - Cross validation (second part)
#' 
#' 
#' 
#' 
#' Supervised machine learning and unsupervised machine learning
#' ===
#' 
#' 1. Classification (supervised machine learning):
#'     - With the class label known, build a model (classifier) to predict a future observation. 
#' 2. Clustering (unsupervised machine learning)
#'     - Without knowing the class label, cluster the data according to their similarity.
#' 
#' Classification (supervised machine learning)
#' ===
#' 
#' ![](../figure/supLearning.jpg)
#' 
#' Clustering (unsupervised machine learning)
#' ===
#' 
#' ![](../figure/UnsupLearning.jpg)
#' 
#' 
#' 
#' Decision Tree (what the result looks like)
#' ===
#' 
## ----------------------------------------------------------------------------------------
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
afit <- rpart(as.factor(svi) ~ . - train, data = prostate)
rpart.plot(afit)

#' 
#' ---
#' 
#' - Target: build up a prediction rule to determine svi = 1 or 0
#' - Top node:
#'   - 0: predicted svi status if a new subject falls into this node
#'   - 0.22: probablity svi = 1
#'   - 100\%: all `r nrow(prostate)` subjects
#' - Bottom left:
#'   - 0: predicted svi status if a new subject falls into this node
#'   - 0.11: probablity svi = 1
#'   - 87\%: 84 subjects fall into this node
#' - Bottom right:
#'   - 1: predicted svi status if a new subject falls into this node
#'   - 0.92: probablity svi = 1
#'   - 13\%: 13 subjects fall into this node
#' 
#' 
#' 
#' 
#' Decision Tree
#' ===
#' 
#' - Decision tree is a supervised learning algorithm (having a pre-defined labels).
#' - It works for both categorical and continuous input and output variables.
#' - In this technique, we split the population or sample into two or more homogeneous sets (or sub-populations) based on the best splitter.
#' 
#' 
#' Motivating example
#' ===
#' 
#' - Sample size n = 30
#' - Variables (features): three variables 
#'     - Gender (Boy/ Girl)
#'     - Class (IX or X)
#'     - Height (5 to 6 ft)
#' - Outcome (labels): 15 out of these 30 play cricket in leisure time.
#' 
#' Purpose: predict whether a student will play cricket in his/her leisure time, based on his/her three variables.
#' 
#' 
#' 
#' Decision tree
#' ===
#' 
#' ![](../figure/motivating.png)
#' 
#' 
#' - Decision tree identifies the most significant variable and it’s value that gives best homogeneous sets of population.
#' 
#' Questions:
#' 
#' - What does the decision tree structure look like?
#' - How to define **homogeneous**?
#' - How to make a prediction for a new person?
#' 
#' 
#' Decision tree structure
#' ===
#' 
#' ![](../figure/Decision_Tree_2.png)
#' 
#' Terminology:
#' 
#' - **Root Node**: It represents entire population or sample and no split yet.
#' - **Splitting**: Dividing a node into two or more sub-nodes.
#' - **Decision Node**: When a sub-node splits into further sub-nodes, then it is called decision node.
#' - **Leaf/ Terminal Node**: Nodes do not split is called Leaf or Terminal node.
#' - **Pruning**: When we remove sub-nodes of a decision node, this process is called pruning. You can say opposite process of splitting.
#' - **Branch / Sub-Tree**: A sub section of entire tree is called branch or sub-tree.
#' - **Parent and Child Node**: A node, which is divided into sub-nodes is called parent node of sub-nodes where as sub-nodes are the child of parent node.
#' 
#' 
#' 
#' 
#' 
#' How does a tree decide where to split?
#' ===
#' 
#' Goodness of split (GOS) criteria
#'   - impurity
#' 
#' Details:
#'   - Selects the split and cutoff which result in most pure subset of nodes.
#'     - explores to split all available nodes.
#'     - explores to split all possible cutoffs for each node.
#' 
#' 
#' ![](../figure/inverseTree.png)
#' 
#' 
#' 
#' Recommended measure of GOS (impurity)
#' ===
#' 
#' - Gini Index
#' - Entropy
#' - Interpretation about impurity:
#'     - 0 indicates pure
#'     - large value indicates impure.
#' 
#' 
#' Gini Index
#' ===
#' 
#' Assume: 
#' 
#' - outcome variable $Y$ is binary ($Y = 0$ or $Y = 1$).
#' - $t$ is a node
#' 
#' $$M_{Gini}(t) = 1 - P(Y = 0 | X \in t)^2 - P(Y = 1 | X \in t)^2$$
#' ![](../figure/motivating.png)
#' 
#' 
#' Gini Index
#' ===
#' 
#' - root: $$M_{Gini}(R) = 1 - 0.5^2 - 0.5^2 = 0.5$$
#' - Gender:Female: $$M_{Gini}(G:F) = 1 - 0.2^2 - 0.8^2 = 0.32$$
#' - Gender:Male: $$M_{Gini}(G:M) = 1 - 0.65^2 - 0.35^2 = 0.455$$
#' - Class:IX: $$M_{Gini}(C:4) = 1 - 0.43^2 - 0.57^2 = 0.4902$$
#' - Class:X: $$M_{Gini}(C:5) = 1 - 0.56^2 - 0.44^2 = 0.4928$$
#' - Height:5.5+: $$M_{Gini}(H:5.5+) = 1 - 0.56^2 - 0.44^2 = 0.4928$$
#' - Height:5.5-: $$M_{Gini}(H:5.5-) = 1 - 0.42^2 - 0.58^2 = 0.4872$$
#' 
#' Goodness of split (GOS) criteria using Gini Index
#' ===
#' 
#' Given an impurity function $M(t)$,
#' the GOS criterion is to find the split $t_L$ and $t_R$ of note $t$ such that the impurity measure is maximally decreased:
#' 
#' $$\arg \max_{t_R, t_L} M(t) - [P(X\in t_L|X\in t) M(t_L) + P(X\in t_R|X\in t) M(t_R)]$$
#' 
#' - If split on Gender:
#' $$M_{Gini}(R) - \frac{10}{30}M_{Gini}(G:F) - \frac{20}{30}M_{Gini}(G:M)
#' = 0.5 - 10/30\times 0.32 - 20/30\times 0.455 = 0.09
#' $$
#' 
#' - If split on Class:
#' $$M_{Gini}(R) - \frac{14}{30}M_{Gini}(C:4) - \frac{16}{30}M_{Gini}(C:5)
#' = 0.5 - 14/30\times 0.4902 - 16/30\times 0.4928 = 0.008
#' $$
#' 
#' - If split on Height:5.5
#' $$M_{Gini}(R) - \frac{12}{30}M_{Gini}(H:5.5-) - \frac{18}{30}M_{Gini}(H:5.5+)
#' = 0.5 - 12/30*0.4872 - 18/30*0.4928 = 0.00944
#' $$
#' 
#' Therefore, we will split based on Gender (maximized decrease).
#' 
#' - Why 5.5? Actually we need to search all possible height cutoffs to select the best cutoff.
#' 
#' 
#' 
#' Entropy 
#' ===
#' 
#' Assume: 
#' 
#' - outcome variable $Y$ is binary ($Y = 0$ or $Y = 1$).
#' - $t$ is a node
#' 
#' $$M_{entropy}(t) =  - P(Y = 0 | X \in t)\log P(Y = 0 | X \in t)
#' - P(Y = 1 | X \in t)\log P(Y = 1 | X \in t)$$
#' 
#' ![](../figure/motivating.png)
#' 
#' 
#' Entropy 
#' ===
#' 
#' - root: $$M_{entropy}(R) = -0.5 \times \log(0.5) -0.5 \times \log(0.5) = 0.6931472$$
#' - Gender:Female: $$M_{entropy}(G:F) = -0.2 \times \log(0.2) -0.8 \times \log(0.8) = 0.5004024$$
#' - Gender:Male: $$M_{entropy}(G:M) = -0.65 \times \log(0.65) -0.35 \times \log(0.35) =  0.6474466$$
#' - Class:IX: $$M_{entropy}(C:4) = -0.43 \times \log(0.43) -0.57 \times \log(0.57) = 0.6833149$$
#' - Class:X: $$M_{entropy}(C:5) = -0.56 \times \log(0.56) -0.44 \times \log(0.44) = 0.6859298$$
#' - Height:5.5+: $$M_{entropy}(H:5.5+) =  -0.56 \times \log(0.56) -0.44 \times \log(0.44) = 0.6859298$$
#' - Height:5.5-: $$M_{entropy}(H:5.5-) = -0.42 \times \log(0.42) -0.58 \times \log(0.58) = 0.680292$$
#' 
#' 
#' Goodness of split (GOS) criteria using entropy
#' ===
#' 
#' Given an impurity function $M(t)$,
#' the GOS criterion is to find the split $t_L$ and $t_R$ of note $t$ such that the impurity measure is maximally decreased:
#' 
#' $$\arg \max_{t_R, t_L} M(t) - [P(X\in t_L|X\in t) M(t_L) + P(X\in t_R|X\in t) M(t_R)]$$
#' 
#' - If split on Gender:
#' $$M_{entropy}(R) - \frac{10}{30}M_{entropy}(G:F) - \frac{20}{30}M_{entropy}(G:M)
#' = 0.6931472 - 10/30\times 0.5004024 - 20/30\times 0.6474466 = 0.09471533
#' $$
#' 
#' - If split on Class:
#' $$M_{entropy}(R) - \frac{14}{30}M_{entropy}(C:4) - \frac{16}{30}M_{entropy}(C:5)
#' = 0.6931472 - 14/30\times 0.6833149 - 16/30\times 0.6859298 = 0.008437687
#' $$
#' 
#' - If split on Height:5.5
#' $$M_{entropy}(R) - \frac{12}{30}M_{entropy}(H:5.5-) - \frac{18}{30}M_{entropy}(H:5.5+)
#' = 0.6931472 - 12/30 \times 0.680292 - 18/30\times 0.6859298 = 0.00947252
#' $$
#' 
#' Therefore, we will split based on Gender (maximized decrease).
#' 
#' 
#' Summary of impurity measurement
#' ===
#' 
#' - Categorical outcome
#'     - Gini Index
#'     - Entropy
#'     - Chi-Square
#' - Continuous outcome
#'     - Reduction in Variance
#' 
#' Complexity for each split:
#' 
#' - $O(np)$
#'   - for each variable, search for all possible cutoffs (n).
#'   - Repeat this for all variables (p of them).
#'   
#' Decision tree
#' ===
#' 
#' 
#' How to make a prediction:
#' 
#' $$\hat{p}_{mk} = \frac{1}{N_m} \sum_{x_i \in t_m} \mathbb{I}(y_i = k)$$
#' 
#' E.g. if a new subject (G:Male, Class:X, H:5.8ft) falls into a terminal node,
#' we just do a majority vote.
#' 
#' 
#' 
#' Regression Trees vs Classification Trees
#' ===
#'  
#' - Regression trees
#'   - dependent variable is continuous
#'   - predictive value at a terminal node: mean response of observation falling in that region
#' $$\hat{c}_{m} = ave(y_i|x_i \in T_m)$$
#' 
#' - Classification trees
#'   - dependent variable is categorical
#'   - predictive value at a terminal node: mode of observations falling in that region
#' 
#' 
#' Prostate cancer example
#' ===
#' 
## ----------------------------------------------------------------------------------------
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
head(prostate)

#' 
#' 
#' Training set and testing set
#' ===
#' 
#' For supervised machine learning, usually we split the data into the training set and the testing set.
#' 
#' - Training set: to build the classifier (build the decision rule in the CART model)
#' - Testing set: evaluate the performance of the classifier from the training set.
#' 
#' 
#' 
#' Apply cart
#' ===
#' 
## ----------------------------------------------------------------------------------------
prostate_train <- subset(prostate, subset = train==TRUE)
prostate_test <- subset(prostate, subset = train==FALSE)

afit <- rpart(as.factor(svi) ~ . - train, data = prostate_train) ## svi as binary outcome
#afit <- rpart(svi ~ . - train, data = prostate_train) ## svi as continuous outcome
afit

#' 
#' Visualize cart result
#' ===
#' 
## ----------------------------------------------------------------------------------------
rpart.plot(afit)

#' 
#' - Top node:
#'   - 0: predicted svi status if a new subject falls into this node
#'   - 0.22: probablity svi = 1
#'   - 100\%: all `r nrow(prostate_train)` subjects
#' - Bottom left:
#'   - 0: predicted svi status if a new subject falls into this node
#'   - 0.06: probablity svi = 1
#'   - 78\%: 52 subjects fall into this node
#' - Bottom right:
#'   - 1: predicted svi status if a new subject falls into this node
#'   - 0.80: probablity svi = 1
#'   - 22\%: 15 subjects fall into this node
#' 
#' 
#' predicting the testing dataset using CART subject
#' ===
#' 
## ----------------------------------------------------------------------------------------
predProb_cart <- predict(object = afit, newdata = prostate_test)
head(predProb_cart)
atable <- table(predictLabel = predProb_cart[,"1"]>0.5, trueLabel = prostate_test$svi)
atable
## accuracy
sum(diag(atable)) / sum(atable)

#' 
#' 
#' Overfitting problem
#' ===
#' 
#' ![](../figure/overfitting.png)
#' 
#' 
#' 
#' 
#' 
#' Pruning the tree (to reduce overfitting)
#' ===
#' 
#' - Brieman et al (1984) proposed a backward node recombinaiton strategy called minimal cost-complexity pruning.
#' - The subtrees are nested sequence:
#' $$T_{max} = T_1 \subset T_2 \subset \ldots \subset T_m$$
#' - The cost-complexity of $T$ is
#' $$C_\alpha (T) = \hat{R}(T) + \alpha |T|,$$
#' where 
#' 
#' - $\hat{R}(T)$ is the error rate estimated based on tree $T$.
#' - $T$ is the tree structure.
#' - $|T|$ is the number of terminal nodes of $T$.
#' - $\alpha$ is a positive complexity parameter that balance between bias and variance (complexity).
#' - $\alpha$ is estimated from cross-validation.
#' - Another way is to set $\alpha = \frac{R(T_i) - R(T_{i-1})}{|T_{i-1}| - |T_{i}|}$
#' 
#' 
#' 
#' Titanic example (Will be on HW)
#' ===
#' 
## ----------------------------------------------------------------------------------------
library(titanic)
dim(titanic_train)
head(titanic_train)

#' 
#' 
#' 
#' Bagging
#' ===
#' 
#' ![](../figure/bagging.png)
#' 
#' 
#' Bagging
#' ===
#' 
#' 1. Create Multiple DataSets:
#'     - Sampling with replacement on the original data
#'     - Taking row and column fractions less than 1 helps in making robust models, less prone to overfitting
#' 2. Build Multiple Classifiers:
#'     - Classifiers are built on each data set.
#' 3. Combine Classifiers:
#'     - The predictions of all the classifiers are combined using a mean, median or mode value depending on the problem at hand.
#'     - The combined values are generally more robust than a single model.
#' 
#' Random forest
#' ===
#' 
#' 
#' ![](../figure/randomForest2.jpg)
#' 
#' 
#' 
#' 
#' 
#' Random forest algorithm
#' ===
#' 
#' ![](../figure/randomForest.jpg)
#' 
#' Assume number of cases in the training set is N, and number of features is M. 
#' 
#' 1. Repeat the following procedures B = 500 times (each time is a CART algorithm):
#'     1. Sample N cases with replacement.
#'     2. Sample m<M features without replacement.
#'     3. Apply the CART algorithm without pruning.
#' 2. Predict new data by aggregating the predictions of the B trees (i.e., majority votes for classification, average for regression).
#' 
#' Random forest
#' ===
#' 
#' ![](../figure/randomForest3.png)
#' 
#' 
#' 
#' Random Forest on prostate cancer example
#' ===
#' 
#' 
## ----------------------------------------------------------------------------------------
library("randomForest")
prostate_train <- subset(prostate, subset = train==TRUE)
prostate_test <- subset(prostate, subset = train==FALSE)

rffit <- randomForest(as.factor(svi) ~ . - train, data = prostate_train)
rffit

#' 
#' 
#' Random Forest on prostate prediction
#' ===
#' 
## ----------------------------------------------------------------------------------------
pred_rf <- predict(rffit, prostate_test)

ctable <- table(pred_rf, trueLabel = prostate_test$svi)
ctable
## accuracy
sum(diag(ctable)) / sum(ctable)



#' 
#' 
#' Random Forest importance score
#' ===
#' 
## ----------------------------------------------------------------------------------------
library(ggplot2)
imp <- importance(rffit)
impData <- data.frame(cov = rownames(imp), importance=imp[,1])

ggplot(impData) + aes(x=cov, y=importance, fill=cov) + geom_bar(stat="identity")

#' 
#' How to calculate Random Forest importance score
#' ===
#' 
#' - Within each tree $t$,
#'   - $\hat{y}_i^{(t)}$: predicted class before permuting
#'   - $\hat{y}_{i,\pi_j}^{(t)}$: predicted class after permuting $x_j$
#' $$VI^{(t)}(x_j) = \frac{1}{|n^{(t)}|} \sum_{i \in n^{(t)}} I(y_i = \hat{y}_i^{(t)}) - 
#' \frac{1}{|n^{(t)}|} \sum_{i \in n^{(t)}} I(y_i = \hat{y}_{i,\pi_j}^{(t)})$$
#' - Raw importance:
#' $$VI(x_j) = \frac{\sum_{t=1}^B VI^{(t)}(x_j)}{B}$$
#' - Scaled improtance: importance() funciton in R: 
#' $$z_j = \frac{VI(x_j)}{\hat{\sigma}/\sqrt{B}}$$
#' 
#' 
#' 
#' Outline for cross validation
#' ===
#' 
#' - cross validation to evaluate machine learning accuracy
#'     - holdout method
#'     - k-fold cross validation
#'     - leave one out cross validation
#' - cross validation to select tuning parameters
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
## ----------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------
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
#' - K-nearest-neighbors (KNN): K (will be introduced on Wednesday)
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
## ----------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------
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
#' References
#' ===
#' 
#' - <https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/>
#' 
#' - <https://www.statistik.uni-dortmund.de/useR-2008/slides/Strobl+Zeileis.pdf>
#' 
#' - Element of stat learning <https://web.stanford.edu/~hastie/ElemStatLearn/>
#' 
