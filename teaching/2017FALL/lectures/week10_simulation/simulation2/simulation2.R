#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Simulation studies 2"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday November 1, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' Outlines
#' ===
#' 
#' 
#' - Further simulations    
#'     - Compare CART, logistic regression, random forest
#'     - Central limit theorm
#'     - Multiple comparison
#'     - Validate a distribution as a mixture of other distributions
#'     - Compare the power of ward test, score test and likelihood ratio test
#' 
#' package setup page
#' ===
#' 
## ------------------------------------------------------------------------
library(MASS)
library(rpart)
library(randomForest)
library(ggplot2)

#' 
#' 
#' Compare machine learning classifier
#' ===
#' 
#' 
#' 
## ------------------------------------------------------------------------
set.seed(32611)
N<-100
d1<-mvrnorm(N, c(1,-1), matrix(c(2, 1, 1, 2), 2, 2))
d2<-mvrnorm(N, c(-1,1), matrix(c(2, 1, 1, 2), 2, 2))
d <- rbind(d1, d2)
label <- c(rep("1", N), rep("2", N))
data_train <- data.frame(label=label, x=d[,1], y=d[,2])
names(data_train)<-c("label", "x", "y")

ggplot(data_train) + aes(x=x,y=y,col=label) + geom_point()

#' 
#' 
#' CART
#' ===
#' 
## ------------------------------------------------------------------------
acart <- rpart(label ~ . ,data=data_train)

avec <- seq(-4,4,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x=z[,1],y=z[,2])
predict_cart <- predict(acart, z, type = "vector")

z_cart <- cbind(region=as.factor(predict_cart), z)
ggplot(z_cart) + aes(x=x,y=y,col=region) + geom_point() + 
  labs(title = "CART boundary")

#' 
#' 
#' Random forest
#' ===
#' 
## ------------------------------------------------------------------------
aRF <- randomForest(label ~ . ,data=data_train)

avec <- seq(-4,4,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x=z[,1],y=z[,2])
predict_RF <- predict(aRF, z)

z_RF <- cbind(region=as.factor(predict_RF), z)
ggplot(z_RF) + aes(x=x,y=y,col=region) + geom_point() + 
  labs(title = "Random forest boundary")

#' 
#' 
#' Logistic regression
#' ===
#' 
## ------------------------------------------------------------------------
aLogistic <- glm(label ~ . ,data=data_train, family = binomial())

avec <- seq(-4,4,0.1)
z <- expand.grid(avec, avec)
z <- data.frame(x=z[,1],y=z[,2])
predict_Logistic <- ifelse(predict(aLogistic, z, type="response")>0.5,2,1)

z_Logistic <- cbind(region=as.factor(predict_Logistic), z)
ggplot(z_Logistic) + aes(x=x,y=y,col=region) + geom_point() + 
  labs(title = "logistic regression boundary")

#' 
#' Compare which classifier works the best in terms of accuracy
#' === 
#' 
#' - Testing dataset
#' 
## ------------------------------------------------------------------------
set.seed(32611)
N_test<-10000
d1_test<-mvrnorm(N_test, c(1,-1), matrix(c(2, 1, 1, 2), 2, 2))
d2_test<-mvrnorm(N_test, c(-1,1), matrix(c(2, 1, 1, 2), 2, 2))
d_test <- rbind(d1_test, d2_test)
label_test <- c(rep("1", N_test), rep("2", N_test))
data_test <- data.frame(label=label_test, x=d_test[,1], y=d_test[,2])
names(data_test)<-c("label", "x", "y")

ggplot(data_test) + aes(x=x,y=y,col=label) + geom_point(alpha = 0.7)

#' 
#' 
#' Prediction accuracy for CART
#' === 
#' 
## ------------------------------------------------------------------------
prediction_test_cart <- predict(acart, data_test, type="vector")
table_test_cart <- table(prediction_test_cart, data_test$label)
table_test_cart
accuracy_test_cart <- sum(diag(table_test_cart))/sum(table_test_cart)
accuracy_test_cart

#' 
#' 
#' Prediction accuracy for random forest
#' === 
#' 
## ------------------------------------------------------------------------
prediction_test_RF <- predict(aRF, data_test)
table_test_RF <- table(prediction_test_RF, data_test$label)
table_test_RF
accuracy_test_RF <- sum(diag(table_test_RF))/sum(table_test_RF)
accuracy_test_RF

#' 
#' Prediction accuracy for logistic regression
#' === 
#' 
## ------------------------------------------------------------------------
prediction_test_logistic <- ifelse(predict(aLogistic, data_test, type="response")>0.5,2,1)
table_test_logistic <- table(prediction_test_logistic, data_test$label)
table_test_logistic
accuracy_test_logistic <- sum(diag(table_test_logistic))/sum(table_test_logistic)
accuracy_test_logistic

#' 
#' Central limit theorm
#' ===
#' 
#' Suppose $\{X_1, X_2, \ldots, X_n\}$ is a sequence of i.i.d. random variables with $E(X_i) = \mu$ and $Var(X_i) = \sigma^2 < \infty$.
#' Then as $n$ approaches infinity, the following random variables converge in distribution to a standarded normal $N(0,1)$
#' 
#' $$\frac{\sqrt{n} (\frac{1}{n} \sum_{i=1}^n X_i - \mu)}{\sigma} \rightarrow N(0, 1)$$
#' 
#' 
#' Poisson distribution
#' ===
#' 
#' - For Poisson distribution with parameter $\lambda$
#' 
#' $$f(k) = \frac{\lambda^k \exp(-\lambda)}{k!}$$
#' 
#' - $\mu = \lambda$
#' - $\sigma^2 = \lambda$
#' 
#' n = 10
#' ===
#' 
## ------------------------------------------------------------------------
n <- 10
lambda <- 5

B <- 5000
collection <- numeric(B)
  
for(b in 1:B){
  set.seed(b)
  x <- rpois(n = n, lambda = lambda)
  collection[b] <- sqrt(n) * (mean(x) - lambda) / sqrt(lambda)
}

hist(collection)

qqnorm(collection)
abline(a = 0, b = 1, col = 2)

shapiro.test(collection)

#' 
#' 
#' n = 40
#' ===
#' 
## ------------------------------------------------------------------------
n <- 40
lambda <- 5

B <- 5000
collection <- numeric(B)
  
for(b in 1:B){
  set.seed(b)
  x <- rpois(n = n, lambda = lambda)
  collection[b] <- sqrt(n) * (mean(x) - lambda) / sqrt(lambda)
}

hist(collection)

qqnorm(collection)
abline(a = 0, b = 1, col = 2)

shapiro.test(collection)

#' 
#' 
#' n = 100
#' ===
#' 
## ------------------------------------------------------------------------
n <- 100
lambda <- 5

B <- 5000
collection <- numeric(B)
  
for(b in 1:B){
  set.seed(b)
  x <- rpois(n = n, lambda = lambda)
  collection[b] <- sqrt(n) * (mean(x) - lambda) / sqrt(lambda)
}

hist(collection)

qqnorm(collection)
abline(a = 0, b = 1, col = 2)

shapiro.test(collection)

#' 
#' Multiple testing
#' ===
#' 
#' - Say you have a set of hypothesis that you want to test simultaneously.
#' - The first idea is to test each hypothesis separately, using a same significance $\alpha$.
#' 
#' - i.e. You have 20 hypothesis to test at a significance level $\alpha = 0.05$. What is the probability of observing at least one significant result just due to chance?
#' 
#' $\begin{aligned}
#' P(\mbox{at least one significant result}) &= 1 - P(\mbox{no significant result}) \\ 
#' & = (1 - 0.05)^{20} \\
#' & \approx 0.64 \\
#' \end{aligned}$
#' 
#' - i.e. In genomic study, say you are tesing 10,000 genes and all of them are non-significant.
#' At $\alpha$ level $0.05$, by chance you will declare 500 sigfinicant genes.
#' 
#' 
#' Bonferroni correction
#' ===
#' 
#' - The Bonferroni correction sets the significance cut-off at $\alpha / n$.
#' - In the previous example with 20 tests and $\alpha = 0.05$, we will only reject a null hypothesis if the p-value is less than 0.0025.
#' 
#' $\begin{aligned}
#' P(\mbox{at least one significant result}) &= 1 - P(\mbox{no significant result}) \\ 
#' & = (1 - 0.0025)^{20} \\
#' & \approx 0.0488 \\
#' \end{aligned}$
#' 
#' - Bonferroni correction tends to be a bit too conservative (and very conservative if the samples are correlated).
#' 
#' Simulation experiment
#' ===
#' 
## ------------------------------------------------------------------------
B <- 1000
n <- 20
alpha <- 0.05

count <- 0
for(b in 1:B){
  set.seed(b)
  ap <- runif(n)
  
  if(any(ap < alpha/n)){
    count <- count + 1
  }
}
count/B


#' 
#' 
#' False discovery rate
#' ===
#' 
#' - raw $\alpha$ level: doesn't control for multiple comparison. False positive by chance. Too loose.
#' - Bonferroni correction: among all tests for samples from null distribution, the probability to falsely reject at least one sample is $\alpha$. Too stringent.
#' - False discover rate (FDR)
#'     - This is defined as the proportion of false positives among all significant results.
#' 
#' ![](../figure/FDRTable.png)
#' 
#' - Control $FDR = FP/R$ within a pre-specified range.
#' 
#' Simulation setting
#' ===    
## ------------------------------------------------------------------------
B1 <- 9000
B2 <- 1000
B <- B1 + B2
n <- 10

SigLabel <- c(rep(F,  B1), rep(T, B2))

pval <- numeric(B)

for(b in 1:B){
    set.seed(b)
    if(b <= B1){
    x <- rnorm(n)
  } else {
    x <- rnorm(n, mean = 2)
  }
  
  pval[b] <- t.test(x=x)$p.value
}

#' 
#' 
#' False discovery rate example
#' ===
## ------------------------------------------------------------------------
qvalue <- p.adjust(pval, "BH")
test_fdr <- qvalue < 0.05
table_fdr <- table(SigLabel, test_fdr)
table_fdr

fpr_fdr <-  sum(test_fdr & !SigLabel) / sum(test_fdr)
fpr_fdr

#' 
#' Comparison - raw p-value
#' ===
#' 
## ------------------------------------------------------------------------
test_raw <- pval < 0.05
table_raw <- table(SigLabel, test_raw)
table_raw

fdr_raw <-  sum(test_raw & !SigLabel) / sum(test_raw)
fnr_raw <-  sum(!test_raw & SigLabel) / sum(!test_raw)
fdr_raw
fnr_raw

#' 
#' Comparison - bonferroni correction
#' ===
## ------------------------------------------------------------------------
test_bonferroni <- pval < 0.05/length(pval)
table_bonferroni <- table(SigLabel, test_bonferroni)
table_bonferroni

fpr_bonferroni <-  sum(test_bonferroni & !SigLabel) / sum(test_bonferroni)
fnr_bonferroni <- sum(!test_bonferroni & SigLabel) / sum(!test_bonferroni)
fpr_bonferroni
fnr_bonferroni

#' 
#' Comparison - false discovery rate
#' ===
## ------------------------------------------------------------------------
qvalue <- p.adjust(pval, "BH")
test_fdr <- qvalue < 0.05
table_fdr <- table(SigLabel, test_fdr)
table_fdr

fpr_fdr <-  sum(test_fdr & !SigLabel) / sum(test_fdr)
fnr_fdr <- sum(!test_fdr & SigLabel) / sum(!test_fdr)
fpr_fdr
fnr_fdr

#' 
#' Validate a distribution as a mixture of other distributions
#' ===
#' 
#' - Bayesian statistics, conjugate prior
#' 
#' - e.g.
#' $$Y|\mu \sim N(\mu, \sigma^2)$$
#' $$\mu \sim N(\mu_0, \sigma_0^2)$$
#' We can conclude the following relationship by some tedious derivation
#' 
#' $$Y \sim N(\mu_0, \sigma^2 + \sigma_0^2)$$
#' 
#' - What if you are not confident about your derivation?
#' 
#' 
#' Simulaitons (try to increase n)
#' ===
#' 
## ------------------------------------------------------------------------
library(ggplot2)
mu0 <- 2
sigma <- 1
sigma0 <- 3

## using derived formula
n <- 1000
y1 <- rnorm(n, mean = mu0, sd = sqrt(sigma^2 + sigma0^2))

## using mixture approach
mu <- rnorm(n, mean = mu0, sd = sigma0)
y2 <- rnorm(n, mean = mu, sd = sigma)

df <- data.frame(x = c(y1, y2),
                 g = gl(2, n))
ggplot(df, aes(x, colour = g)) + geom_density()



#' 
#' 
#' compare ward test, score test and likelihood ratio test (in one dimension)
#' ===
#' 
#' - Poisson distribution
#' 
#' $$f(y; \lambda) = \frac{\exp(-\lambda) \lambda^y}{y!}$$
#' - Likelihood function
#' 
#' $$L(\lambda; y_1, \ldots, y_n) = \prod_{i=1}^n \frac{\exp(-\lambda) \lambda^{y_i}}{y_i!}$$
#' - log likelihood function
#' 
#' $\begin{aligned}
#' l(\lambda; y_1, \ldots, y_n) & = \sum_{i=1}^n log \frac{\exp(-\lambda) \lambda^{y_i}}{y_i!} \\ 
#' & \propto \sum_i y_i \log \lambda - n \lambda \\ 
#' \end{aligned}$
#' 
#' 
#' ---
#' 
#' - Score function
#' 
#' $\begin{aligned}
#' s(\lambda) & = \frac{1}{n}\frac{\partial l(\lambda)}{\partial \lambda} \\
#' & = \frac{1}{n} \frac{\sum_i y_i}{\lambda} - 1 \\
#' \end{aligned}$
#' 
#' - Fisher's information
#' 
#' $\begin{aligned}
#' I(\lambda) & = - E(\frac{1}{n}\frac{\partial^2 l(\lambda)}{\partial \lambda^2}) \\
#' & = E(\frac{1}{n} \frac{\sum_i y_i}{\lambda^2} ) \\
#' & = \frac{1}{\lambda} \\
#' \end{aligned}$
#' 
#' 
#' 
#' compare ward test, score test and likelihood ratio test (in one dimension)
#' ===
#' 
#' - Wald test
#' 
#' $$n (\lambda - \lambda_0)^2 \hat{I}(\lambda) \sim \chi^2(1)$$
#' 
#' 
#' - Score test
#' 
#' $$n S(\lambda_0)^2 I(\lambda_0)^{-1}  \sim \chi^2(1)$$
#' 
#' - likelihood ratio test
#' 
#' $$- 2 \log \{ \frac{\sup_{\lambda = \lambda_0}L(\lambda)}{\sup_{\lambda \in \Lambda}L(\lambda)} \} \sim \chi^2(1)$$
#' 
#' compare ward test, score test and likelihood ratio test
#' ===
#' 
#' ![](../figure/asyTest.gif)
#' 
#' 
#' Simulation, check if these tests achieve advertised size
#' ===
#' 
#' - $\lambda_0 = 5$
#' - vary sample sizes
#'     - n = 20
#' 
#' Simulation -- Wald test
#' ===
#' 
#' 
## ------------------------------------------------------------------------
lambda0 = 5
n <- 20 ## increase
B <- 1000
alpha <- 0.05

count <- 0
for(b in 1:B){
  set.seed(b)
  y <- rpois(n, lambda = lambda0)
  lambda_hat <- mean(y)
  T_ward <- n * (lambda_hat - lambda0)^2 * (1/lambda_hat)
  if(pchisq(T_ward, df = 1, lower.tail = F) < alpha){
    count <- count + 1
  }
}
count / B

#' 
#' 
#' Simulation -- Score test
#' ===
#' 
## ------------------------------------------------------------------------
lambda0 = 5
n <- 20 ## increase
B <- 1000
alpha <- 0.05

count <- 0
for(b in 1:B){
  set.seed(b)
  y <- rpois(n, lambda = lambda0)
  lambda_hat <- mean(y)
  s <- lambda_hat/lambda0 - 1
  T_score <- n * s^2 * lambda0
  if(pchisq(T_score, df = 1, lower.tail = F) < alpha){
    count <- count + 1
  }
}
count / B

#' 
#' 
#' Simulation -- likelihood ratio test
#' ===
#' 
## ------------------------------------------------------------------------
lambda0 = 5
n <- 20 ## increase
B <- 1000
alpha <- 0.05

count <- 0
for(b in 1:B){
  set.seed(b)
  y <- rpois(n, lambda = lambda0)
  lambda_hat <- mean(y)
  l <- n * (lambda_hat * log(lambda_hat) - lambda_hat)
  l0 <- n * (lambda_hat * log(lambda0) - lambda0)
  T_lr <- -2*(l0 - l)
  if(pchisq(T_lr, df = 1, lower.tail = F) < alpha){
    count <- count + 1
  }
}
count / B

#' 
#' compare ward test, score test and likelihood ratio test -- using large sample
#' ===
#' 
## ------------------------------------------------------------------------
lambda0 = 5
n <- 20 ## increase
B <- 1000
alpha <- 0.05

count_wald <- 0
count_score <- 0
count_lr <- 0
for(b in 1:B){
  set.seed(b)
  y <- rpois(n, lambda = lambda0)
  lambda_hat <- mean(y)
  s <- lambda_hat/lambda0 - 1
  l <- n * (lambda_hat * log(lambda_hat) - lambda_hat)
  l0 <- n * (lambda_hat * log(lambda0) - lambda0)

  T_ward <- n * (lambda_hat - lambda0)^2 * (1/lambda_hat)
  T_score <- n * s^2 * lambda0
  T_lr <- -2*(l0 - l)

  if(pchisq(T_ward, df = 1, lower.tail = F) < alpha) count_wald <- count_wald + 1
  if(pchisq(T_score, df = 1, lower.tail = F) < alpha) count_score <- count_score + 1
  if(pchisq(T_lr, df = 1, lower.tail = F) < alpha) count_lr <- count_lr + 1
}

count_wald / B
count_score / B
count_lr / B


#' 
#' Will be on HW
#' ===
#' 
#' Compare the power of ward test, score test and likelihood ratio test
#' 
## ------------------------------------------------------------------------
knitr::purl("simulation2.rmd", output = "simulation2.R ", documentation = 2)

