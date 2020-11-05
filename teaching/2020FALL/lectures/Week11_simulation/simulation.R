#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Simulation study"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday November 9, 2020"
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
#' - Compare estimators
#'     - mean estimators
#'     - linear regression estimators
#' - Hypothesis testing
#'     - size
#'     - power
#'     - confidence intercal
#' - Permutation test
#' - Multiple testing
#'     - Bonferroni correction
#'     - False discovery rate
#' 
#' 
#' What is a simulation study?
#' ===
#' 
#' Simulation (a.k.a Monte Carlo simulation):
#' 
#' - Rely on repeated random samplings to obtain numerical results.
#' 
#' 
#' Why we need simulation studies
#' 
#' - After we propose a statistical method, we want to validate the method using simulation so people can use it with confidence.
#' - Exact analytical derivations of properties are rarely possible.
#' 
#' 
#' Common questions simulation study can answer
#' ===
#' 
#' - How does an estimator compare to its competing estimators on the basis of bias, mean square error, efficiency, etc.?
#' - Does a procedure for constructing a confidence interval for a parameter achieve the advertised nominal level of coverage?
#' - Does a hypothesis testing procedure attain the nominal (advertised) level of significance or size?
#' - How to estimate power from simulation study?
#' 
#' 
#' 
#' How to approximate
#' ===
#' 
#' Typical Monte Carlo simulation involves the following procedures:
#' 
#' 1. Generate $B$ independent datasets $(1, 2, \ldots, B)$ under conditions of interest.
#' 2. Compute the numerical value of an estimator/test statistic $T$ from each dataset.
#'     Then we have $T_1, T_2, \ldots, T_b, \ldots, T_B$.
#' 3. If $B$ is large enough, summary statistics across $T_1, \ldots, T_B$ should be good approximations to the true properties of the estimator/test statistic.
#' 
#' Illustration:
#' 
#' $X \sim f(\theta)$, $T = T(X)$, where $T$ is an estimator, or a procedure
#' 
#' ![](../figure/simulationOutline.png)
#' 
#' 
#' 
#' 
#' Example 1, Compare estimators (normal distribution)
#' ===
#' 
#' - Assume $X \sim N(\mu, \sigma^2)$ with $\mu = 1$ and $\sigma^2 = 1$.
#' - We want to compare three estimators for the mean $\mu$ of this distribution based on a random sample $X_1, X_2, \ldots, X_n$.
#'     - Sample Mean $T^{mean}$
#'     - Sample Median $T^{median}$
#'     - Sample 20\% trimmed mean $T^{mean20\%}$
#' 
#' Note: 20\% trimmed mean indicates average value after deleting first 10\% quantile and last 10\% quantile.
#' 
#' E.g. 1,2,...,10
#' 
#' - mean = mean(1:10)
#' - mean20\% = mean(2:9)
#' 
#' Evaluation criteria
#' ===
#' 
#' k: estimator index, including: mean, median, mean20\%
#' 
#' - $T^{(k)}$: estimator of interest
#' - Bias: $|E({T}^{(k)}) - \mu|$
#' - variance: $E[{T}^{(k)} - E({T}^{(k)})]^2$
#' - **Mean squared error**: $MSE = Bias^2 + vairance$
#'     - MSE is a commonly used criteria for evaluating estimators.
#'     - $MSE = E[{T}^{(k)} - \mu]^2$
#' - Relative efficiency 
#'     - For any consistent estimator $T^{(1)}$, $T^{(2)}$, (i.e., $E(T^{(1)}) = E(T^{(2)}) = \mu$)
#'     - $RE(T^{(2)}, T^{(1)}) = \frac{MSE(T^{(1)})}{MSE(T^{(2)})}$
#'     - is the relative efficiency of estimator 2 to estimator 1.
#'     - $RE < 1$ means estimator 1 is preferred.
#'     
#' $T^{mean}$
#' ===
#' 
#' 
#' $$T^{mean} = \frac{1}{n}\sum_{i=1}^n X_i$$
#' 
#' - Bias: $|E({T}^{(mean)}) - \mu| = |\frac{1}{n}\sum_{i=1}^n \mu - \mu| = 0$
#' - Variance: $E[{T}^{(mean)} - E({T}^{(mean)})]^2 = \frac{\sigma^2}{n}$
#' - MSE: $MSE = Bias^2 + vairance = \frac{\sigma^2}{n}$
#' 
#' Can be calculated analytically
#' 
#' 
#' $T^{median}$
#' ===
#' 
#' 
#' $$T^{median} = Median_{i=1}^n X_i$$
#' 
#' - Bias: $|E({T}^{(median)}) - \mu| =0?$
#' - Variance: $E[{T}^{(median)} - E({T}^{(median)})]^2 =?$
#' - MSE: $MSE = Bias^2 + vairance =?$
#' 
#' Variance is hard to be calculated analytically
#' 
#' $T^{mean20\%}$
#' ===
#' 
#' Can be calculated
#' 
#' $$T^{mean20\%} = \frac{1}{0.8n}\sum_{i=0.1n+1}^{0.9n} X_i$$
#' 
#' - Bias: $| E({T}^{(mean20\%)}) - \mu| =0?$
#' - Variance: $E[{T}^{(mean20\%)} - E({T}^{(mean20\%)})]^2 =?$
#' - MSE: $MSE = Bias^2 + vairance =?$
#' 
#' Variance is hard to be calculated analytically
#' 
#' 
#' Evaluation via simulation
#' ===
#' 
#' No analytic solution, we can evaluate using simulation studies.
#' 
#' - Algorithm:
#'     1. Sample $n$ samples from $f$ (i.e., $N(\mu, \sigma^2)$)
#'     2. Calculate ${T}^{(k)}$, where $k$ is the estimator index
#'     3. Repeat Step 1,2 B times to obtain $T_1^{(k)}, \ldots, T_B^{(k)}$
#'     4. Evaluate the property of ${T}^{(k)}$ using the Monte Carlo samples $T_1^{(k)}, \ldots, T_B^{(k)}$.
#'     
#' - $\bar{T}^{(k)} = \frac{1}{B}\sum_{b=1}^B T_b^{(k)}$
#' - $\widehat{Bias} = |\bar{T}^{(k)} - \mu|$
#' - $\widehat{vairance} = \frac{1}{B - 1} \sum_{b=1}^B (T_b^{(k)} - \bar{T}^{(k)})^2$
#' - Mean squared error: $\widehat{MSE} = \widehat{Bias}^2 + \widehat{vairance}$
#' - Relative efficiency 
#'     - For any consistent estimator $T^{(1)}$, $T^{(2)}$, (i.e., $E(T^{(1)}) = E(T^{(2)}) = \mu$)
#'     - $\widehat{RE} = \frac{\widehat{MSE}(T^{(1)})}{\widehat{MSE}(T^{(2)})}$
#'     - is the relative efficiency of estimator 2 to estimator 1.
#'     - $\widehat{RE} < 1$ means estimator 1 is preferred.
#' 
#' Simulation experiment
#' ===
#' 
## ------------------------------------------------------------------------
mu <- 1; B <- 1000; n <- 20

Ts <- matrix(0,nrow=B,ncol=3)
colnames(Ts) <- c("mean", "median", "mean 20% trim")


for(b in 1:B){
  set.seed(b)
  x <- rnorm(n, mean = mu)
  
  Ts[b,1] <- mean(x)
  Ts[b,2] <- median(x)
  Ts[b,3] <- mean(x, trim = 0.1)
}

bias <- abs(apply(Ts,2,function(x) mean(x) - mu))
vars <- apply(Ts,2,var)

MSE <- bias^2 + vars
RE <- MSE[1]/MSE

comparisonTable <- rbind(bias, vars, MSE, RE)
colnames(comparisonTable) <- c("mean", "median", "mean 20% trim")
comparisonTable

#' 
#' 
#' Difference between simulation and real data application
#' ===
#' 
#' - In simulation
#'     - Know the underlying truth (e.g., $\mu$)
#'     - Can evaluate the performance of our estimator/procedure (e.g., $T$) by comparing to the underlying truth
#' - In real data application
#'     - There is no underlying truth
#' 
#' 
#' 
#' Example 2, Compare estimators (exponential distribution)
#' ===
#' 
#' - Assume $X \sim Exp(\mu)$ with $\mu = 1$.
#' $$f(x) = \exp(-x), x\ge0$$
#' - We want to compare three estimators for the mean $\mu$ of this distribution based on a random sample $X_1, X_2, \ldots, X_n$.
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

Ts <- matrix(0,nrow=B,ncol=3)
colnames(Ts) <- c("mean", "median", "mean 20% trim")

for(b in 1:B){
  set.seed(b)
  x <- rexp(n, rate = mu)
  
  Ts[b,1] <- mean(x)
  Ts[b,2] <- median(x)
  Ts[b,3] <- mean(x, trim = 0.1)
}

bias <- abs(apply(Ts,2,function(x) mean(x) - mu))
vars <- apply(Ts,2,var)

MSE <- bias^2 + vars
RE <- MSE[1]/MSE

comparisonTable <- rbind(bias, vars, MSE, RE)
colnames(comparisonTable) <- c("mean", "median", "mean 20% trim")
comparisonTable

#' 
#' Example 3, Compare estimators (Cauchy distribution)
#' ===
#' 
#' - Assume $X \sim Cauchy(\mu,\gamma)$ with $\mu = 0$ and $\gamma = 1$.
#' $$f(x; \mu, \gamma) = \frac{1}{\pi (1 + x^2)}$$
#' 
#' - We want to compare three estimators for the mean $\mu$ of this distribution based on a random sample $X_1, X_2, \ldots, X_n$.
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

Ts <- matrix(0,nrow=B,ncol=3)
colnames(Ts) <- c("mean", "median", "mean 20% trim")

for(b in 1:B){
  set.seed(b)
  x <- rcauchy(n, location  = mu)
  
  Ts[b,1] <- mean(x)
  Ts[b,2] <- median(x)
  Ts[b,3] <- mean(x, trim = 0.1)
}

bias <- abs(apply(Ts,2,function(x) mean(x) - mu))
vars <- apply(Ts,2,var)

MSE <- bias^2 + vars
RE <- MSE[1]/MSE

comparisonTable <- rbind(bias, vars, MSE, RE)
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
#' $$\hat{\beta}^{ols} = \arg \min_\beta \frac{1}{2}\|y - X\beta\|^2_2$$
#' 
#' - Modern statistics care about both bias and variance of an estimator instead of only unbiased estimator.
#' 
#' - Lasso estimator
#' 
#' $$\hat{\beta}^{lasso} = \arg \min_\beta \frac{1}{2} \|y - X\beta\|^2_2 + \lambda \|\beta\|_1$$
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
#'     2. Calculate $\hat{\beta}^{ols}_b \in \mathbb{R}^6$
#' - Calculate $\hat{\bar{\beta}}^{ols} = \frac{1}{B}\sum_{b=1}^B \hat{\beta}^{ols}_b$
#' - Calculate bias $bias(\beta, \hat{\bar{\beta}}^{ols})^2 = \sum_{j=0}^5 bias(\beta_j, \hat{\bar{\beta}}^{ols}_j)^2$
#' - Calculate variance: $var(\hat{\beta}^{ols}_j) = \frac{1}{B-1} \sum_{b=1}^B (\hat{\beta_j}^{ols}_b - \hat{\bar{\beta}}_j^{ols})^2$ ,
#'     $var_T(\hat{\beta}^{ols}) = \sum_{j=0}^5 var(\hat{\beta}^{ols}_j)$
#' - $MSE(\hat{\beta}^{ols}) = bias(\beta, \hat{\beta}^{ols})^2 + var_T(\hat{\beta}^{ols})$
#' 
#' 
#' For simulation of lasso 
#' ===
#' 
#' - Set $B = 1000$
#' - For $b (1 \le b \le B)$, given a specific tuning parameter,
#'     1. simulate $Y_b = Y_0 + N(0,3^2)$
#'     2. Calculate $\hat{\beta}^{lasso}_b \in \mathbb{R}^6$
#' - Calculate $\hat{\bar{\beta}}^{lasso} = \frac{1}{B}\sum_{b=1}^B \hat{\beta}^{lasso}_b$
#' - Calculate bias $bias(\beta, \hat{\bar{\beta}}^{lasso})^2 = \sum_{j=0}^5 bias(\beta_j, \hat{\bar{\beta}}^{lasso}_j)^2$
#' - Calculate variance: $var(\hat{\beta}^{lasso}_j) = \frac{1}{B-1} \sum_{b=1}^B (\hat{\beta_j}^{lasso}_b - \hat{\bar{\beta}}_j^{lasso})^2$ ,
#'     $var_T(\hat{\beta}^{lasso}) = \sum_{j=0}^5 var(\hat{\beta}^{lasso}_j)$
#' - $MSE(\hat{\beta}^{lasso}) = bias(\beta, \hat{\beta}^{lasso})^2 + var_T(\hat{\beta}^{lasso})$
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
#' Tuning parameter selection for lasso?
#' ===
#' 
#' 1. Split the data to part A and part B.
#' 2. Using part A, find the tuning parameter such that the MSE is minimized.
#' 3. Evaluate the performance (with the tuning parameter in [2]) using part B.
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
#' ===
#' 
#' - Test $H_0: \mu = \mu_0$ against $H_A: \mu \neq \mu_0$
#' - To evaluate whether size/level of test achieves advertised $\alpha$ under $H_0$
#'     - Generate data under $H_0: \mu = \mu_0$ and calculate the proportion of rejections of $H_0$
#'     - Proportion of rejecting $H_0$ approximately equals to $\alpha$
#' 
#' 
#' Consider using the following simulation setting
#' 
#' - n = 20 (number of samples)
#' - B = 1000 (number of simulations)
#' - alpha = 0.05
#' - two.sided test
#' 
#' Example 5, t.test() size
#' === 
#' 
#' - Test $H_0: \mu = 0$ against $H_A: \mu \neq 0$
#' 
#' 
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
#' - Test $H_0: \mu = 0$ against $H_A: \mu \neq 0$
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
#' 
#' - Test $H_0: \mu = 0$ against $H_A: \mu \neq 0$
#' 
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
#' 
#' Compare these test procedures
#' ===
#' 
#' - If several tests has the same size, how to compare which test is better? 
#'     - Compare power
#' - To evaluate power
#'     - Generate data under some alternatives, $\mu \neq \mu_0$ (i.e.,$\mu = \mu_1$) and calculate the proportion of rejections of $H_0$
#' 
#' 
#' Example 8, t.test() power
#' === 
#' 
#' - Test $H_0: \mu = 0$ against $H_A: \mu = 1$
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
#' - Test $H_0: \mu = 0$ against $H_A: \mu = 1$
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
#' - Test $H_0: \mu = 0$ against $H_A: \mu = 1$
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
#' In general, the larger sample sizes, the larger power.
#' 
#' Example 10, Validate the result using R function
#' ===
#' 
## ------------------------------------------------------------------------
alpha <- 0.05; mu=0; mu1 <- 1; n <- 20; B <- 1000
pwr.t.test(n = 10, d = mu1, sig.level = alpha, type = "one.sample", alternative = "two.sided")

#' 
#' Summary: Size and Power
#' ===
#' 
#' - Size of a test: 
#'   Generate data under $H_0$, at alpha level 0.05, the proportation of rejecting $H_0$ is the size of a test
#' 
#' - Power of a test: 
#'   Generate data under $H_A$, at alpha level 0.05, the proportation of rejecting $H_0$ is the power of a test
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
#' Example 11, Confidence interval
#' ===
#' 
#' - $\mu = 1$
#' - $n=10, i = 1, \ldots, n$
#' - $X_i \sim N(\mu, 1)$
#' - $95\%$ confidence interval of $\mu$: 
#' $(\bar{X} - qt(0.975, n-1) \times \frac{\hat{\sigma}}{\sqrt{n}}, \bar{X} + qt(0.975, n-1) \times \frac{\hat{\sigma}}{\sqrt{n}})$
#' 
## ------------------------------------------------------------------------
mu <- 1
B <- 1000
n <- 10

x <- rnorm(n, mean=mu)
c(mean(x) - qt(0.975, n-1) * sd(x) / sqrt(n), mean(x) + qt(0.975, n-1) * sd(x) / sqrt(n))

#' 
#' ---
## ------------------------------------------------------------------------
t.test(x)

#' 
#' ---
#' 
#' - Evaluate the performance of this CI
#'     - Whether the procedure will trap the true parameter 95% chances
#' 
## ------------------------------------------------------------------------
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
#' 
#' 
#' Example 12, Confidence interval (using normal approximation, n=100)
#' ===
#' 
#' - CI by normal approximation
#' - Performance for large $n$, (e.g., $n=100$)
#' 
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
#' Example 12, Confidence interval (using normal approximation, n=10)
#' ===
#' 
#' - behavior in small sample sizes (e.g., $n=10$)
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
#' Multiple testing
#' ===
#' 
#' - Say you have a set of hypothesis that you want to test simultaneously.
#' - The first idea is to test each hypothesis separately, using a same significance $\alpha$.
#' - i.e.,You have 20 hypothesis to test at a significance level $\alpha = 0.05$. What is the probability of observing at least one significant result just due to chance?
#' 
#' $\begin{aligned}
#' P(\mbox{at least one significant result}) &= 1 - P(\mbox{no significant result}) \\ 
#' & = 1 - (1 - 0.05)^{20} \\
#' & \approx 0.64 \\
#' \end{aligned}$
#' 
#' - i.e.,In genomic study, say you are tesing 10,000 genes and all of them are non-significant.
#' At $\alpha$ level $0.05$, by chance you will declare 500 sigfinicant genes.
#' 
#' 
#' Bonferroni correction
#' ===
#' 
#' - Bonferroni correction controls the familywise error rate.
#'   - Bonferroni corrected p-value is the pvalue for at least one significant result
#' - The Bonferroni correction sets the significance cut-off at $\alpha / n$.
#' - In the previous example with 20 tests and $\alpha = 0.05$, we will only reject a null hypothesis if the p-value is less than 0.0025.
#' 
#' $\begin{aligned}
#' P(\mbox{at least one significant result}) &= 1 - P(\mbox{no significant result}) \\ 
#' & = 1 - (1 - 0.0025)^{20} \\
#' & \approx 0.0488 \\
#' \end{aligned}$
#' 
#' - Bonferroni correction tends to be a bit too conservative (and very conservative if the samples are correlated).
#' 
#' Simulation experiment
#' ===
#' 
#' Under the null hypothesis, the p-values follow $U(0,1)$.
#' 
## ------------------------------------------------------------------------
B <- 1000
n <- 20 ## number of tests
N <- 15 ## number of samples per test
alpha <- 0.05

count <- 0
for(b in 1:B){
  set.seed(b)
  ax <- replicate(n, rnorm(N),simplify = F)
  ap <- sapply(ax,function(avec) t.test(avec)$p.value)
  
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
#'     - Also known as Benjamini-Hochberg correction
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

fpr_raw <-  sum(test_raw & !SigLabel) / sum(test_raw)
fnr_raw <-  sum(!test_raw & SigLabel) / sum(!test_raw)
fpr_raw
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
#' 3. Keep B small at first
#'     - Test and refine code until you are sure everything is working correctly before carrying out final production runs
#'     - Get an idea of how long it takes to process one data set
#'     
#' 4. Set a different seed for each run and keep records
#'     - Ensure simulation runs are independent
#'     - Runs may be replicated if necessary
#' 
#' 5. Document your code
#' 
#' Reference
#' ===
#' 
#' - <https://www.stat.nus.edu.sg/~stazjt/teaching/ST2137/lecture/lec%2011.pdf>
#' - <http://www4.stat.ncsu.edu/~davidian/st810a/simulation_handout.pdf>
#' 
#' 
