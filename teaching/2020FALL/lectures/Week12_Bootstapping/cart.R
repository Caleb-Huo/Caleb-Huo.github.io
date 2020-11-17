#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Wednesday November 18, 2020"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Decision Tree
#' ---
#' 
#' 
#' Outline
#' ===
#' 
#' - Classification and regression tree
#' - Random forest
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
## ----------------------------------------------------------------------------------------------
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
#' 
#' - Decision tree explores to split all available nodes.
#' - Decision tree explores to split all possible cutoffs for each node.
#' - Selects the split and cutoff which result in most homogeneous sub-nodes.
#' 
#' 
#' ![](../figure/inverseTree.png)
#' 
#' 
#' 
#' Recommended measure of impurity 
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
## ----------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------
prostate_train <- subset(prostate, subset = train==TRUE)
prostate_test <- subset(prostate, subset = train==FALSE)

afit <- rpart(as.factor(svi) ~ . - train, data = prostate_train) ## svi as binary outcome
#afit <- rpart(svi ~ . - train, data = prostate_train) ## svi as continuous outcome
afit

#' 
#' Visualize cart result
#' ===
#' 
## ----------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------
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
#' Will be on homework
#' ===
#' 
#' Apply random forest on Titanic dataset to predict survival.
#' 
#' - What are the importance factors?
#' - How does the performance compare to CART.
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
