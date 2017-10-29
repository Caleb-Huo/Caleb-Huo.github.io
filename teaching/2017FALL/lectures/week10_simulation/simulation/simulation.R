#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Simulation studies"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday October 30, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' Outlines
#' ===
#' 
#' - Introduction
#' - Examples
#'     - Compare estimators for mean
#'     - Compare OLS and lasso estimators
#'     - Hypothesis testing, size
#'     - Hypothesis testing, power
#'     - Hypothesis testing, Confidence interval
#'     - Hypothesis testing, Confidence interval, finite sample sizes
#'     
#' - Further simulations    
#'     
#' What is a simulation study?
#' ===
#' 
#' Simulation:
#' 
#' - A numerical technique for conducting experiments on the computer.
#' - It involves random sampling from probability distributions.
#' 
#' Rationale for simulation study
#' ===
#' 
#' - After we propose a statistical method, we want to validate the method using simulation so people can use it with confidence.
#' - Exact analytical derivations of properties are rarely possible.
#' - Large sample approximations to properties are often possible.
#'     - But what happens in the finite sample size case?
#'     - And what happens when assumptions are violated?
#' 
#' 
#' Common questions simulation study can answer
#' ===
#' 
#' - Is an estimator consistent with infinite samples?
#'     - If so, Is an estimator biased in finite samples?
#'     - Is it still consistent under departures from assumptions?
#' - How does an estimator compare to its competing estimators on the basis of bias, mean square error, efficiency, etc.?
#' - Does a procedure for constructing a confidence interval for a parameter achieve the advertised nominal level of coverage?
#' - Does a hypothesis testing procedure attain the nominal (advertised) level of significance or size?
#' - How to estimate power from simulation study?
#' 
#' 
#' Monte Carlo simulation approximation
#' ===
#' 
#' - An test statistic or estimator has a true sampling distribution.
#' - Ideally, we could want to know this true sampling distribution in order to address the issues on the previous slide.
#' - But derivation of the true sampling distribution is not tractable.
#' - Hence we approximate the sampling distribution of an estimator or a test statistic under a particular set of conditions.
#' 
#' How to approximate (Draw a picture to illustrate)
#' ===
#' 
#' Typical Monte Carlo simulation involves the following procedure:
#' 
#' 1. Generate $B$ independent datasets $(1, 2, \ldots, B)$ under conditions of interest.
#' 2. Compute the numerical value of an estimator/test statistic $T$ from each dataset.
#'     Then we have $T_1, T_2, \ldots, T_b, \ldots, T_B$.
#' 3. If $B$ is large enough, summary statistics across $T_1, \ldots, T_B$ should be good approximations to the true properties of the estimator/test statistic.
#' 
#' Example:
#' 
#' Consider an estimator for a mean parameter $\theta$:
#' 
#' - $T_b$ is the value of $T$ from the $b^{th}$ data, $(b = 1, \ldots,  B)$.
#' - The mean of all $T_b$'s is an estimate of the true mean.
#' 
#' We will have more concrete examples later.
#' 
#' 
#' Example 1, Compare estimator (normal distribution)
#' ===
#' 
#' - Assume $X \sim N(\mu, \sigma^2)$ with $\mu = 1$ and $\sigma^2 = 1$.
#' - We want to compare three estimators for the mean $\mu$ of this distribution based on a random sample $Y_1, Y_2, \ldots, Y_n$.
#'     - Sample Mean $T^{mean}$
#'     - Sample Median $T^{median}$
#'     - Sample 20\% trimmed mean $T^{mean20\%}$
#' 
#' Note: 20\% trimmed mean indicates average value after deleting first 10\% quantile and last 10\% quantile.
#' 
#' E.g. 3,2,4,7,1,6,5,8,9,10
#' 
#' - mean = mean(1:10)
#' - mean20\% = mean(2:9)
#' 
#' Evaluation criteria
#' ===
#' 
#' k: mean, median, mean20\%
#' 
#' - $T^{(k)}$: estimator of interest
#' - Bias: $E({T}^{(k)}) - \mu$
#' - variance: $E(({T}^{(k)} - \mu)^2)$
#' - **Mean squared error**: $MSE = Bias^2 + vairance$
#'     - MSE is a commonly used criteria for evaluating estimators.
#' - Relative efficiency 
#'     - For any consistent estimator $T^{(1)}$, $T^{(2)}$, (i.e. $E(T^{(1)}) = E(T^{(2)}) = \mu$)
#'     - $RE = \frac{MSE(T^{(1)})}{MSE(T^{(2)})}$
#'     - is the relative efficiency of estimator 2 to estimator 1.
#'     - $RE < 1$ means estimator 1 is preferred.
#'     
#' $T^{mean}$
#' ===
#' 
#' 
#' $$T^{mean} = \frac{1}{n}\sum_{i=1}^n Y_i$$
#' 
#' - Bias: $E({T}^{(mean)}) - \mu = \frac{1}{n}\sum_{i=1}^n \mu - \mu = 0$
#' - Variance: $E(({T}^{(mean)} - \mu)^2) = \frac{\sigma^2}{n}$
#' - MSE: $MSE = Bias^2 + vairance = \frac{\sigma^2}{n}$
#' 
#' Can be calculated analytically
#' 
#' 
#' $T^{median}$
#' ===
#' 
#' 
#' $$T^{median} = Median_{i=1}^n Y_i$$
#' 
#' - Bias: $E({T}^{(median)}) - \mu =0?$
#' - Variance: $E(({T}^{(median)} - \mu)^2) =?$
#' - MSE: $MSE = Bias^2 + vairance =?$
#' 
#' Variance is hard to be calculated analytically
#' 
#' $T^{mean20\%}$
#' ===
#' 
#' Can be calculated
#' 
#' $$T^{mean20\%} = \frac{1}{0.8n}\sum_{i=0.1n+1}^{0.9n} Y_i$$
#' 
#' - Bias: $E({T}^{(mean20\%)}) - \mu =0?$
#' - Variance: $E(({T}^{(mean20\%)} - \mu)^2) =?$
#' - MSE: $MSE = Bias^2 + vairance =?$
#' 
#' Variance is hard to be calculated analytically
#' 
#' 
#' Evaluation via simulation
#' ===
#' 
#' Since sometimes it is hard to evaluate the estimator analytically,
#' we can evaluate use simulation studies.
#' 
#' - algorithm:
#'     1. Sample $n$ subject from $f$ (i.e. $N(\mu, \sigma^2)$)
#'     2. Calculate ${T}^{(k)}$
#'     3. Repeat Step 1,2 B times to obtain $T_1^{(k)}, \ldots, T_B^{(k)}$
#'     4. Evaluate the property of ${T}^{(k)}$ using the Monte Carlo samples $T_1^{(k)}, \ldots, T_B^{(k)}$.
#'     
#' - $\bar{T}^{(k)} = \sum_{b=1}^B T_b^{(k)}$
#' - $\widehat{Bias} = \bar{T}^{(k)} - \mu$
#' - $\widehat{vairance} = \frac{1}{B - 1} \sum_{b=1}^B (T_b^{(k)} - \bar{T}^{(k)})$
#' - Mean squared error: $\widehat{MSE} = \widehat{Bias}^2 + \widehat{vairance}$
#' - Relative efficiency 
#'     - For any consistent estimator $T^{(1)}$, $T^{(2)}$, (i.e. $E(T^{(1)}) = E(T^{(2)}) = \mu$)
#'     - $\widehat{RE} = \frac{\widehat{MSE}(T^{(1)})}{\widehat{MSE}(T^{(2)})}$
#'     - is the relative efficiency of estimator 2 to estimator 1.
#'     - $\widehat{RE} < 1$ means estimator 1 is preferred.
#' 
#' Simulation experiment
#' ===
#' 
## ------------------------------------------------------------------------
mu <- 1; B <- 1000; n <- 20

T1 <- numeric(B); T2 <- numeric(B); T3 <- numeric(B)

for(b in 1:B){
  set.seed(b)
  x <- rnorm(n, mean = mu)
  
  T1[b] <- mean(x)
  T2[b] <- median(x)
  T3[b] <- mean(x, trim = 0.1)
}

bias_T1 <- abs(mean(T1) - mu); bias_T2 <- abs(mean(T2) - mu); bias_T3 <- abs(mean(T3) - mu)
bias <- c(bias_T1, bias_T2, bias_T3)

var_T1 <- var(T1); var_T2 <- var(T2); var_T3 <- var(T3)
vars <- c(var_T1, var_T2, var_T3)

MSE1 <- bias_T1^2 + var_T1; MSE2 <- bias_T2^2 + var_T2; MSE3 <- bias_T3^2 + var_T3
MSE <- c(MSE1, MSE2, MSE3)

RE <- MSE1/MSE

comparisonTable <- rbind(bias, vars, MSE, RE)
colnames(comparisonTable) <- c("mean", "median", "mean 20% trim")
comparisonTable

#' 
#' 
#' Example 2, Compare estimator (exponential distribution)
#' ===
#' 
#' - Assume $X \sim Exp(\mu)$ with $\mu = 1$.
#' $$f(x) = \exp(-x), x\ge0$$
#' - We want to compare three estimators for the mean $\mu$ of this distribution based on a random sample $Y_1, Y_2, \ldots, Y_n$.
#'     - Sample Mean $T^{mean}$
#'     - Sample Median $T^{median}$
#'     - Sample 20\% trimmed mean $T^{mean20\%}$
#' 
#' 
#' Simulation experiment
#' ===
#' 
## ------------------------------------------------------------------------
mu <- 1; B <- 1000; n <- 20

T1 <- numeric(B); T2 <- numeric(B); T3 <- numeric(B)

for(b in 1:B){
  set.seed(b)
  x <- rexp(n, rate = mu)
  
  T1[b] <- mean(x)
  T2[b] <- median(x)
  T3[b] <- mean(x, trim = 0.1)
}

bias_T1 <- abs(mean(T1) - mu); bias_T2 <- abs(mean(T2) - mu); bias_T3 <- abs(mean(T3) - mu)
bias <- c(bias_T1, bias_T2, bias_T3)

var_T1 <- var(T1); var_T2 <- var(T2); var_T3 <- var(T3)
vars <- c(var_T1, var_T2, var_T3)

MSE1 <- bias_T1^2 + var_T1; MSE2 <- bias_T2^2 + var_T2; MSE3 <- bias_T3^2 + var_T3
MSE <- c(MSE1, MSE2, MSE3)

comparisonTable <- rbind(bias, vars, MSE)
colnames(comparisonTable) <- c("mean", "median", "mean 20% trim")
comparisonTable

#' 
#' Example 3, Compare estimator (Cauchy distribution)
#' ===
#' 
#' - Assume $X \sim Cauchy(\mu,\gamma)$ with $\mu = 0$ and $\gamma = 1$.
#' $$f(x; \mu, \gamma) = \frac{1}{\pi (1 + x^2)}$$
#' 
#' - We want to compare three estimators for the mean $\mu$ of this distribution based on a random sample $Y_1, Y_2, \ldots, Y_n$.
#'     - Sample Mean $T^{mean}$
#'     - Sample Median $T^{median}$
#'     - Sample 20\% trimmed mean $T^{mean20\%}$
#' 
#' 
#' Simulation experiment
#' ===
#' 
## ------------------------------------------------------------------------
mu <- 0; B <- 1000; n <- 20

T1 <- numeric(B); T2 <- numeric(B); T3 <- numeric(B)

for(b in 1:B){
  set.seed(b)
  x <- rcauchy(n, location  = mu)
  
  T1[b] <- mean(x)
  T2[b] <- median(x)
  T3[b] <- mean(x, trim = 0.1)
}

bias_T1 <- abs(mean(T1) - mu); bias_T2 <- abs(mean(T2) - mu); bias_T3 <- abs(mean(T3) - mu)
bias <- c(bias_T1, bias_T2, bias_T3)

var_T1 <- var(T1); var_T2 <- var(T2); var_T3 <- var(T3)
vars <- c(var_T1, var_T2, var_T3)

MSE1 <- bias_T1^2 + var_T1; MSE2 <- bias_T2^2 + var_T2; MSE3 <- bias_T3^2 + var_T3
MSE <- c(MSE1, MSE2, MSE3)

comparisonTable <- rbind(bias, vars, MSE)
colnames(comparisonTable) <- c("mean", "median", "mean 20% trim")
comparisonTable

#' 
#' 
#' 
#' UMVUE
#' ===
#' 
#' - Uniformly minimum-variance unbiased estimator (UMVUE) is an unbiased estimator that has lower variance than any other unbiased estimator for all possible values of the parameter.
#' 
#' - Blue: the best linear unbiased estimator (BLUE) of the coefficients is given by the ordinary least squares (OLS) estimator, 
#' 
#' $$\hat{\beta}^{ols} = \arg \min_\beta \|y - X\beta\|^2_2$$
#' 
#' - Modern statistics care about both bias and variance of an estimator instead of only unbiased estimator.
#' 
#' - Lasso estimator
#' 
#' $$\hat{\beta}^{lasso} = \arg \min_\beta \|y - X\beta\|^2_2 + \lambda \|\beta\|_1$$
#' 
#' Example 4, evaluate bias, variance and MSE of OLS, lasso
#' ===
#' 
#' Model setup: 
#' 
#' - $\beta_0 = 1$, $\beta_1 = 3$, $\beta_2 = 5$, $\beta_3 = \beta_4 = \beta_5 = 0$
#' - $X_1, \ldots, X_5 \sim N(0,1)$
#' - $Y_0 = \beta_0 + X_1 \beta_1 + X_2 \beta_2 +\ldots,  X_5 \beta_5$
#' - $Y = Y_0 + N(0,3^2)$
#' - $n = 30$
#' 
#' Goal:
#' 
#' - Wants to evaluate bias, variance and MSE of OLS, lasso estimators
#' - $bias(\beta, \hat{\beta})^2 = \sum_{j=0}^5 bias(\beta_j, \hat{\beta}_j)^2$
#' - $var_T(\hat{\beta}) = \sum_{j=0}^5 var(\hat{\beta}_j)$
#' - $MSE(\hat{\beta}) = bias(\beta, \hat{\beta})^2 + var_T(\hat{\beta})$
#' 
#' For simulation of OLS (ordinary least square)
#' ===
#' 
#' - Set $B = 1000$
#' - For $b (1 \le b \le B)$
#'     1. simulate $Y_b = Y_0 + N(0,3^2)$
#'     2. Calculate $\hat{\beta}^{ols}_b$
#'     3. Calculate $\hat{\bar{\beta}}^{ols} = \frac{1}{B}\sum_{b=1}^B \hat{\beta}^{ols}_b$
#'     4. Calculate bias $bias(\beta, \hat{\bar{\beta}}^{ols})^2 = \sum_{j=0}^5 bias(\beta_j, \hat{\bar{\beta}}^{ols}_j)^2$
#'     5. Calculate variance: $var(\hat{\beta}) = \frac{1}{B-1} (\hat{\beta}^{ols}_b - \hat{\bar{\beta}}^{ols})^2$ ,
#'     $var_T(\hat{\beta}) = \sum_{j=0}^5 var(\hat{\beta}_j)$
#'     6. $MSE(\hat{\beta}) = bias(\beta, \hat{\beta})^2 + var_T(\hat{\beta})$
#' 
#' 
#' For simulation of lasso 
#' ===
#' 
#' - Set $B = 1000$
#' - For $b (1 \le b \le B)$, given a specific tuning parameter,
#'     1. simulate $Y_b = Y_0 + N(0,3^2)$
#'     2. Calculate $\hat{\beta}^{lasso}_b$
#'     3. Calculate $\hat{\bar{\beta}}^{lasso} = \frac{1}{B}\sum_{b=1}^B \hat{\beta}^{lasso}_b$
#'     4. Calculate bias $bias(\beta, \hat{\bar{\beta}}^{lasso})^2 = \sum_{j=0}^5 bias(\beta_j, \hat{\bar{\beta}}^{lasso}_j)^2$
#'     5. Calculate variance: $var(\hat{\beta}) = \frac{1}{B-1} (\hat{\beta}^{lasso}_b - \hat{\bar{\beta}}^{lasso})^2$ ,
#'     $var_T(\hat{\beta}) = \sum_{j=0}^5 var(\hat{\beta}_j)$
#'     6. $MSE(\hat{\beta}) = bias(\beta, \hat{\beta})^2 + var_T(\hat{\beta})$
#' - Repeat this procedure for all tuning parameters
#' 
#' Simulation experiment
#' ===
#' 
## ------------------------------------------------------------------------
library(lars)

beta0 <- 1; beta1 <- 3; beta2 <- 5
beta3 <- beta4 <- beta5 <- 0
beta <- c(beta0,beta1,beta2,beta3,beta4,beta5)
p <- 5; n <- 30
set.seed(32611)
X0 <- matrix(rnorm(p*n),nrow=n)
X <- cbind(1, X0) ## X fixed

Y0 <- X %*% as.matrix(beta, ncol=1)

B <- 1000
beta_ols <- replicate(n = B, expr = list())
beta_lasso <- replicate(n = B, expr = list())

for(b in 1:B){
  set.seed(b)

  error <- rnorm(length(Y0), sd = 3)
  Y <- Y0 + error
  
  abeta_ols <- lm(Y~X0)$coefficients
  beta_ols[[b]] <- abeta_ols
  
  alars <- lars(x = X, y = Y, intercept=F)
  beta_lasso[[b]] <- alars
}


#' 
#' Simulation experiment (continue 2)
#' ===
#' 
## ------------------------------------------------------------------------

beta_ols_hat <- Reduce("+",beta_ols)/length(beta_ols)
beta_ols_bias <- sqrt(sum((beta_ols_hat - beta)^2))
beta_ols_variance <-  sum(Reduce("+", lapply(beta_ols,function(x) (x - beta_ols_hat)^2))/(length(beta_ols) - 1))
beta_ols_MSE <- beta_ols_bias + beta_ols_variance

as <- seq(0,10,0.1)
lassoCoef <- lapply(beta_lasso, function(alars){
  alassoCoef <- t(coef(alars, s=as, mode="lambda"))
  alassoCoef
} )

beta_lasso_hat <- Reduce("+",lassoCoef)/length(lassoCoef)
beta_lasso_bias <- sqrt(colSums((beta_lasso_hat - beta)^2))
beta_lasso_variance <-  colSums(Reduce("+", lapply(lassoCoef,function(x) (x - beta_lasso_hat)^2))/(length(beta_ols) - 1))
beta_lasso_MSE <- beta_lasso_bias + beta_lasso_variance

#' 
#' Simulation experiment (continue 3)
#' === 
#' 
## ------------------------------------------------------------------------
par(mfrow=c(2,2))
plot(x = as, y = beta_lasso_bias, type = "l", xlab="lambda", main="bias")
abline(h = beta_ols_bias, lty=2)
legend("topleft", legend = c("lasso", "OLS"), col = 1, lty = c(1,2))

plot(x = as, y = beta_lasso_variance, col=4, type = "l", xlab="lambda", main="variance")
abline(h = beta_ols_variance, lty=2, col=4)
legend("bottomleft", legend = c("lasso", "OLS"), col = 4, lty = c(1,2))

plot(x = as, y = beta_lasso_MSE, col=2, type = "l", xlab="lambda", main="MSE")
abline(h = beta_ols_MSE, lty=2, col=2)
legend("topleft", legend = c("lasso", "OLS"), col = 2, lty = c(1,2))


plot(x = as, y = beta_lasso_MSE, ylim = c(min(beta_lasso_bias), max(beta_lasso_MSE)), 
     type="n", xlab="lambda", main="MSE = bias^2 + variance")
lines(x = as, y = beta_lasso_MSE, type = "l", col=2)
lines(x = as, y = beta_lasso_variance, col=4)
lines(x = as, y = beta_lasso_bias, col=1)
legend("bottomright", legend = c("bias", "variance", "MSE"), col = c(1,4,2), lty = c(1))

#' 
#' 
#' Hypothesis testing (one-sided)
#' ===
#' 
#' ![](../figure/hypothesisOnesided.png)
#' 
#' Hypothesis testing (two-sided)
#' ===
#' 
#' ![](../figure/hypothesisTwosided.png)
#' 
#' 
#' 
#' Does a hypothesis testing procedure attain the nominal (advertised) level of significance or size?
#' ===
#' 
#' Consider the following tests:
#' 
#' - Parametric: t.test()
#' - Non-Parametric: wilcox.test()
#' 
#' Simulation for sizes of hypothesis tests
#' 
#' - Test $H_0: \mu = \mu_0$ against $H_A: \mu \neq \mu_0$
#' - To evaluate whether size/level of test achieves advertised $\alpha$
#'     - Generate data under $H_0: \mu = \mu_0$ and calculate the proportion of rejections of $H_0$
#' - Approximation the true probability of rejecting $H_0$ when it is true
#'     - Proportion should be approximately equal to $\alpha$
#' 
#' 
#' Consider using the following simulation setting
#' 
#' - n = 20
#' - B = 1000
#' - alpha = 0.05
#' - two.sided test
#' 
#' Example 5, t.test() size
#' === 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; n <- 20; B <- 1000

counts <- 0
for(b in 1:B){
  set.seed(b)
  
  x <- rnorm(n, mean=mu)
  atest <- t.test(x)
  
  if(atest$p.value < alpha){
    counts <- counts + 1
  }
}

counts/B

#' 
#' 
#' Example 6, wilcox.test() size
#' === 
#' 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; n <- 20; B <- 1000

counts <- 0
for(b in 1:B){
  set.seed(b)
  
  x <- rnorm(n, mean=mu)
  atest <- wilcox.test(x)
  
  if(atest$p.value < alpha){
    counts <- counts + 1
  }
}

counts/B

#' 
#' 
#' Example 7, t.test() only use first 10 samples size
#' === 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; n <- 20; B <- 1000

counts <- 0
for(b in 1:B){
  set.seed(b)
  
  x <- rnorm(n, mean=mu)[1:10]
  atest <- t.test(x)
  
  if(atest$p.value < alpha){
    counts <- counts + 1
  }
}

counts/B

#' 
#' Compare these test procedures
#' ===
#' 
#' - If several tests has the same size, how to compare which test is better? - Compare power
#' - To evaluate power
#'     - Generate data under some alternatives, $\mu \neq \mu_0$ (i.e. $\mu = \mu_1$) and calculate the proportion of rejections of $H_0$
#' - Approximate the true probability of rejecting $H_0$ when the alternative is true.
#' 
#' 
#' Example 8, t.test() power
#' === 
#' 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; mu1 <- 1; n <- 20; B <- 1000

counts <- 0
for(b in 1:B){
  set.seed(b)
  
  x <- rnorm(n, mean=mu1)
  atest <- t.test(x)
  
  if(atest$p.value < alpha){
    counts <- counts + 1
  }
}
counts / B

#' 
#' Example 8, Validate the result using R function
#' ===
#' 
## ------------------------------------------------------------------------
library(pwr)
alpha <- 0.05; mu=0; mu1 <- 1; n <- 20; B <- 1000
pwr.t.test(n = n, d = mu1, sig.level = alpha, type = "one.sample", alternative = "two.sided")

#' 
#' Example 9, wilcox.test() power
#' === 
#' 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; mu1 <- 1; n <- 20; B <- 1000

counts <- 0
for(b in 1:B){
  set.seed(b)
  
  x <- rnorm(n, mean=mu1)
  atest <- wilcox.test(x)
  
  if(atest$p.value < alpha){
    counts <- counts + 1
  }
}
counts / B

#' 
#' 
#' Example 10, t.test() only use first 10 samples power
#' === 
#' 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; mu1 <- 1; n <- 20; B <- 1000

counts <- 0
for(b in 1:B){
  set.seed(b)
  
  x <- rnorm(n, mean=mu1)[1:10]
  atest <- t.test(x)
  
  if(atest$p.value < alpha){
    counts <- counts + 1
  }
}
counts / B

#' 
#' Example 10, , Validate the result using R function
#' ===
#' 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; mu1 <- 1; n <- 20; B <- 1000
pwr.t.test(n = 10, d = mu1, sig.level = alpha, type = "one.sample", alternative = "two.sided")

#' 
#' 
#' Confidence interval
#' ===
#' 
#' A $1 - \alpha$ confidence interval for parameter $\mu$ is an interval $C_n = (a,b)$ where 
#' $a = a(X_1, \ldots, X_n)$ and $b = b(X_1, \ldots, X_n)$ are functions of the data such that
#' $$P(\mu \in C_n) \ge 1 - \alpha$$
#' In other words, $(a,b)$ traps $\mu$ with probability $1 - \alpha$.
#' We call $1 - \alpha$ the coverage of the confidence interval.
#' 
#' Interpretation:
#' 
#' - A confidence interval is not a probability statement about $\mu$ since $\mu$ is a fixed quantity, not a random variable.
#' - if I repeat the experiment over and over, the interval will contain the parameter $\mu$ 95 percent of the time.
#' 
#' 
#' Example 11, Confidence interval ($\mu$ can be random)
#' ===
#' 
#' - $\mu = 1$
#' - $n=10, i = 1, \ldots, n$
#' - $X_i \sim N(\mu, 1)$
#' - $95\%$ confidence interval: $(\bar{X} - qt(0.975, n-1) \times \frac{\hat{\sigma}}{\sqrt{n}}, \bar{X} + qt(0.975, n-1) \times \frac{\hat{\sigma}}{\sqrt{n}})$
#' 
## ------------------------------------------------------------------------
mu <- 1
B <- 1000
n <- 10

x <- rnorm(n, mean=mu)
c(mean(x) - qt(0.975, n-1) * sd(x) / sqrt(n), mean(x) + qt(0.975, n-1) * sd(x) / sqrt(n))
t.test(x)

counts <- 0
for(b in 1:B){
  set.seed(b)

  x <- rnorm(n, mean=mu)
  atest <- t.test(x)
  
  if(atest$conf.int[1] < mu & mu < atest$conf.int[2]){
    counts <- counts + 1
  }
}

counts/B

#' 
#' 
#' Example 12, Confidence interval (using normal approximation, n=10)
#' ===
#' 
#' Is the Confidence interval accurate in finite samples?
#' 
#' - $\mu = 1$
#' - $n=10, i = 1, \ldots, N$
#' - $X_i \sim N(\mu, 1)$
#' - $95\%$ confidence interval: $(\bar{X} - 1.96 \times \frac{\hat{\sigma}}{\sqrt{n}}, \bar{X} + 1.96 \times \frac{\hat{\sigma}}{\sqrt{n}})$
#' 
#' 
## ------------------------------------------------------------------------
set.seed(32611)
B <- 1000 ## number of test
counts <- 0 ## number of accepted tests
n <- 10
mu <- 1
alpha <- 0.05
for(b in 1:B){
  x <- rnorm(n, mean=mu, sd = 1)
  lb <- mean(x) - qnorm(alpha/2, lower.tail = F) * sd(x)/sqrt(n)
  ub <- mean(x) + qnorm(alpha/2, lower.tail = F) * sd(x)/sqrt(n)
  if(lb < mu & ub > mu){
    counts <- counts + 1
  }
}
print(counts/B)

#' 
#' 
#' Example 12, Confidence interval (using normal approximation, n=100)
#' ===
#' 
#' - behavior in large sample
#' 
#' - $\mu = 1$
#' - $n=100, i = 1, \ldots, N$
#' - $X_i \sim N(\mu, 1)$
#' - $95\%$ confidence interval: $(\bar{X} - 1.96 \times \frac{\hat{\sigma}}{\sqrt{n}}, \bar{X} + 1.96 \times \frac{\hat{\sigma}}{\sqrt{n}})$
#' 
#' 
## ------------------------------------------------------------------------
set.seed(32611)
B <- 1000 ## number of test
counts <- 0 ## number of accepted tests
n <- 100
mu <- 1
alpha <- 0.05
for(b in 1:B){
  x <- rnorm(n, mean=mu, sd = 1)
  lb <- mean(x) - qnorm(alpha/2, lower.tail = F) * sd(x)/sqrt(n)
  ub <- mean(x) + qnorm(alpha/2, lower.tail = F) * sd(x)/sqrt(n)
  if(lb < mu & ub > mu){
    counts <- counts + 1
  }
}
print(counts/B)

#' 
#' 
#' 
#' 
#' Principle for simulation study
#' ===
#' 
#' 1. A Monte Carlo simulation is just like any other experiment
#'     - Factors that are of interest to vary in the experiment: sample size n, distribution of the data, magnitude of variation...
#'     - Each combination of factors is a separate simulation, so that many factors can lead to very large number of combinations and thus number of simulations (time consuming)
#'     - Results must be recorded and saved in a systematic way
#'     - Don't only choose factors favorable to a method you have developed!
#'     - Sample size B (number of data sets) must deliver acceptable precision.
#' 
#' 2. Save everything!
#'     - Save the individual estimates in a file and then analyze
#'     - as opposed to computing these summaries and saving only them
#'     - Critical if the simulation takes a long time to run!
#'     
#' 3. Keep S small at first
#'     - Test and refine code until you are sure everything is working correctly before carrying out final production runs
#'     - Get an idea of how long it takes to process one data set
#'     
#' 4. Set a different seed for each run and keep records
#'     - Ensure simulation runs are independent
#'     - Runs may be replicated if necessary
#' 
#' 5. Document your code
#' 
#' 
#' Further simulation topics
#' ===
#' 
#' - model assumption is violated
#' - central limit theorem
#' - False discovery rate
#' - machine learning
#' - validate a distribution as a mixture of other distributions.
#' - compare ward test, score test and likelihood ratio test
#' 
#' Reference
#' ===
#' 
#' - <https://www.stat.nus.edu.sg/~stazjt/teaching/ST2137/lecture/lec%2011.pdf>
#' - <http://www4.stat.ncsu.edu/~davidian/st810a/simulation_handout.pdf>
#' 
## ------------------------------------------------------------------------
knitr::purl("simulation.rmd", output = "simulation.R ", documentation = 2)

#' 
