#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Wednesday Nov 29, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: LDA and SVM
#' ---
#' 
#' 
#' Outline
#' ===
#' 
#' - Linear discriminant analysis (LDA)
#' - Support vector machine (SVM)
#' 
#' 
#' 
#' Supervised machine learning
#' ===
#' 
#' - Classification (supervised machine learning):
#'     - With the class label known, learn the features of the classes to predict a future observation. 
#'     - The learning performance can be evaluated by the prediction error rate.
#' 
#' Classification (supervised machine learning)
#' ===
#' 
#' ![](../figure/supLearning.jpg)
#' 
#' Classification (supervised machine learning)
#' ===
#' 
#' - Have learned so far:
#'     - Logistic regression
#'     - KNN
#'     - CART
#'     - Random Forest
#' - Will learn today:
#'     - **LDA**
#'     - **SVM**
#' - Didn't learn this one but it is very popular:
#'     - *Deep learning*
#' 
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
data_train <- data.frame(label=label, x=d[,1], y=d[,2])
names(data_train)<-c("label", "x", "y")

ggplot(data_train) + aes(x=x,y=y,col=label) + geom_point() +  stat_ellipse()

#' 
#' - Intuition:
#'     - We estimate the Gaussian density function for each class, and for a future testing sample, the classifier is "select the class with larger density function value".
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
#' & = \arg \max_k \frac{f(Y=k)f(X|Y=k)}{f(x)} \\
#' & = \arg \max_k f(Y=k)f(X|Y=k) \\
#' & = \arg \max_k \log f(Y=k) + \log f(X|Y=k) \\
#' & = \arg \max_k \log f(Y=k) - \frac{1}{2} \log |\Sigma_k| -  \frac{1}{2} (X - \mu_k)^\top \Sigma_k^{-1} (X - \mu_k) \\
#' & = \arg \min_k - 2 \log f(Y=k) + \log |\Sigma_k| + (X - \mu_k)^\top \Sigma_k^{-1} (X - \mu_k)
#' \end{aligned}$
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
#' You can see the decision boundary is linear and passes the center between $\frac{\mu_l + \mu_m}{2}$.
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
z <- data.frame(x=z[,1],y=z[,2])
predict_lda <- predict(alda, z)$class

z_lda <- cbind(region=as.factor(predict_lda), z)
ggplot(z_lda) + aes(x=x,y=y,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x,y=y,col=label)) +
  labs(title = "LDA boundary")

#' 
#' 
#' 
#' Extension: DLDA
#' ===
#' 
#' Further assume diagonal covariance matrix 
#' $\Sigma_l = \Sigma_m = \Sigma = 
#' \begin{pmatrix}
#' \sigma_1^2 &  \ldots & 0\\ 
#' \ldots & \ldots  & \ldots\\ 
#' 0 & \ldots  & \sigma_p^2
#' \end{pmatrix}$ 
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
data_train <- data.frame(label=label, x=d[,1], y=d[,2])
names(data_train)<-c("label", "x", "y")

ggplot(data_train) + aes(x=x,y=y,col=label) + geom_point() +  stat_ellipse()

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
z <- data.frame(x=z[,1],y=z[,2])
predict_qda <- predict(aqda, z)$class

z_qda <- cbind(region=as.factor(predict_qda), z)
ggplot(z_qda) + aes(x=x,y=y,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x,y=y,col=label)) +
  labs(title = "QDA boundary")

#' 
#' Extension: DQDA
#' ===
#' 
#' Further assume diagonal covariance matrix for $\Sigma_l$ and $\Sigma_m$ respectively.
#' 
#' 
#' Support vector machine (SVM)
#' ===
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
data_train <- data.frame(label=label, x=d[,1], y=d[,2])
names(data_train)<-c("label", "x", "y")

p <- ggplot(data_train) + aes(x=x,y=y,col=label) + geom_point()
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
#' $$f(x) = w^\top x$$
#' 
#' - red dots
#' $$w^\top x \le -1$$
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
#' 
#' - Lagrange function:
#' 
#' $$L(w, \alpha) = \frac{1}{2} \|w\|_2^2 - \sum_i \alpha_i [y_i (w^\top x_i ) - 1]$$
#' By KKT condition, $\frac{\partial L}{\partial w}$ need to be 0 at the optimun solution.
#' 
#' $$\frac{\partial L}{\partial w_i} = w_i - \alpha_i y_i x_i = 0$$
#' $$w_i = \alpha_i y_i x_i$$
#' 
#' Plugging terms back into Lagrange function $L$:
#' $$\max_\alpha  - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i$$
#' Subject to $\alpha_i \ge 0$
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
#' - When $\alpha_i = 0$, no restriction on $y_i w^\top x_i - 1$.
#' - When $\alpha_i > 0$, $y_i w^\top x_i = 1$, which are called support vectors.
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
data_train <- data.frame(label=label, x=d[,1], y=d[,2])
names(data_train)<-c("label", "x", "y")

asvm <- svm(label ~ . ,data=data_train, kernel = "linear")

avec <- seq(-6,6,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x=z[,1],y=z[,2])
predict_svm <- predict(asvm, z)

z_svm <- cbind(region=as.factor(predict_svm), z)
ggplot(z_svm) + aes(x=x,y=y,col=region) + geom_point(alpha = 0.4)  + geom_point(data = data_train, aes(x=x,y=y,col=label)) +
  labs(title = "SVM boundary")

#' 
#' 
#' Property of SVM
#' ===
#' 
#' $$w_i = \alpha_i y_i x_i$$
#' 
#' - Weight vector $w$ is weighted linear combination of samples.
#' - Since $\alpha_i \ne 0$ only if the samples are on the margin, so only the margin samples contribute to $w$.
#' - This is quadratic programming so optimization is not too hard.
#' 
#' $$\max_\alpha  - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j + \sum_i \alpha_i$$
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
data_train <- data.frame(label=label, x=d[,1], y=d[,2])
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
#' ![](../figure/svmHigh.png)
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
data_train <- data.frame(label=label, x=d[,1], y=d[,2])
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
#' Generate R code
#' ===
#' 
## ------------------------------------------------------------------------
knitr::purl("LDAandSVM.rmd", output = "LDAandSVM.R ", documentation = 2)

