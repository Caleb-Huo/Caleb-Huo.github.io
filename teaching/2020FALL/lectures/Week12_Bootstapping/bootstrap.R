#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Bootstrapping"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday November 16, 2020"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' Outline
#' ===
#' 
#' - Bootstrapping
#'     - Bootstrapping variance, confidence interval
#'     - non-parametric Bootstrapping, parametric Bootstrapping
#' 
#' 
#' International Prize in Statistics Awarded to Bradley Efron, for his contribution in Bootstrapping (annouced in 11/12/2018)
#' 
#' - Permutation test
#' 
#' Bootstrapping (Motivating example 1, variance of "mean estimator")
#' ===
#' 
#' - Given $X_1, \ldots, X_n$ samples from $N(\mu, \sigma^2)$. How to estimate the mean of this distribution, $\hat{\mu}$? 
#'     - Use $\frac{1}{n} \sum_{i=1}^n X_i$, by weak law of large number:
#' $$\hat{\mu} = \frac{1}{n} \sum_{i=1}^n X_i \rightarrow E(X_i)$$
#' - How to estimate the variance of $\hat{\mu}$? 
#'     - Similarly use weak law of large number.
#' $$Var(\hat{\mu}) = Var(\frac{1}{n} \sum_{i=1}^n X_i) = \frac{1}{n^2} \sum_{i=1}^n Var(X_i) = \frac{1}{n} \sigma^2 $$
#' 
#' $$\hat{Var}(\hat{\mu}) =\frac{1}{n} \hat{\sigma}^2 = \frac{1}{n(n-1)} \sum_{i=1}^n (X_i - \bar{X})^2$$
#' 
#' Bootstrapping (Motivating example 2, variance of "median estimator")
#' ===
#' 
#' - Given $X_1, \ldots, X_n$ samples from $N(\mu, \sigma^2)$. How to estimate the median of this distribution, $T_n$? 
#'     - Use $T_n = median_{i=1}^n(X_i)$ be a statistic, where $T_n$ is median of $X_1, \ldots, X_n$.
#' - How to estimate $V(T_n)$, the variance of $T_n$? 
#'     - Seems difficult.
#' 
#' 
#' 
#' Aim of bootstrapping: how to estimate variance
#' ===
#' 
#' - Suppose $X_1, \ldots, X_n \sim F$ where $F$ is a distribution.
#' - Let $T_n = g(X_1, \ldots, X_n)$ be a statistic, where $T_n$ is a function of the data.
#' - What is the variance of $T_n$, $V_F(T_n)$?
#' 
#' Variance of "median estimator"
#' ===
#' 
#' - Given $X_1, \ldots, X_n$ samples from $N(\mu_0, \sigma_0^2)$, where both $\mu_0$ and $\sigma_0^2$ are known. How to estimate the median of this distribution, $T_n$? 
#'     - Let $T_n = median_{i=1}^n(X_i)$ be a statistic, where $T_n$ is median of $X_1, \ldots, X_n$.
#' - How to estimate the variance of $T_n$? 
#'     - We can use Monte Carlo simulation method to estimate the variance of $T_n$.
#' 
#' Monte Carlo method
#' 
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n \sim N(\mu_0,\sigma_0^2)$
#'     - Compute $T^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{T}_n = \frac{1}{B} \sum_{b=1}^B T^{(b)}_n$
#' - $\hat{V}_F (T_n) =  \frac{1}{B - 1} \sum_{b=1}^B \{T^{(b)}_n - \bar{T}_n \}^2$
#' - By law of large number, $\hat{V}_F (T_n) \rightarrow V_F(T_n)$
#' 
#' Variance of "median estimator", Monte Carlo method
#' ===
#' 
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
mu <- 1
sigma <- 1
n <- 100
B <- 1000
Ts <- numeric(B)

for(b in 1:B){
  set.seed(b)
  ax <- rnorm(n, mu, sigma)
  Ts[b] <- median(ax)
}

varTest <- var(Ts)
print(varTest)

#' 
#' 
#' - What if we don't know $\mu$, and $\sigma^2$?
#'   - We only have observed data.
#' 
#' 
#' Emperical distribution
#' ===
#' 
#' - $F = P(X \le x)$ is a distribution function.
#' - We can estimate $F$ with the **empirical distribution function**
#' $F_n$, the cdf that puts mass $1/n$ at each data point $X_i$.
#' $$F_n (x) = \frac{1}{n} \sum_{i = 1}^n I(X_i \le x)$$
#' where
#' \begin{equation}
#'     I(X_i \le x) =
#'     \begin{cases}
#'       1, & \text{if}\ X_i \le x \\
#'       0, & \text{if}\ X_i > x
#'     \end{cases}
#'   \end{equation}
#' 
#' 
#' Empirical process (visualization)
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
library(ggplot2)
n <- 1000
df <- data.frame(x = c(rnorm(n, 0, 1)))
base <- ggplot(df, aes(x)) + stat_ecdf()
base + stat_function(fun = pnorm, colour = "red") + xlim(c(-3,3))

#' 
#' Empirical process
#' ===
#' 
#' Empirical distribution is close to the underlying distribution
#' 
#' - Glivenko-Cantelli Theorem
#' $$\sup_x | F_n(x) - F(x)| \rightarrow 0$$
#'   
#' - Dvoretzky-Kiefer-Wolfowitz inequality, for any $\varepsilon > 0$
#' $$P(\sup_x | F_n(x) - F(x)| > \varepsilon) \le 2\exp(-2n\varepsilon^2)$$
#' 
#' 
#' 
#' Variance of the "median estimator", Bootstrapping method 
#' ===
#' 
#' Instead of drawing samples from the underlying distribution $F \sim N(\mu, \sigma^2)$,
#' we draw from the empirical distribution $F_n$
#' 
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n \sim$ ~~N(mu,sigma^2)~~ $F_n$
#'     - Compute $T^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{T}_n = \frac{1}{B} \sum_{b=1}^B T^{(b)}_n$
#' - $\hat{V}_{F_n} (T_n) =  \frac{1}{B-1} \sum_{b=1}^B \{T^{(b)}_n - \bar{T}_n \}^2$
#' 
#' 
#' How to sample from the empirical distribution?
#' ===
#' 
#' - **Drawing $X_1^*, \ldots, X_n^*$ from $F_n$ is equivalent to draw $n$ observations, with replacement from the original data $\{X_1, \ldots, X_n\}$**.
#' - Therefore, Bootstrapping sampling is also described as resampling data.
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n \sim \{ X_1, \ldots, X_n \}$ with replacement.
#'     - Compute $T^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{T}_n = \frac{1}{B} \sum_{b=1}^B T^{(b)}_n$
#' - $\hat{V}_{F_n} (T_n) =  \frac{1}{B-1} \sum_{b=1}^B \{T^{(b)}_n - \bar{T}_n \}^2$
#' 
#' Variance of "median estimator", Bootstrapping method 
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
mu <- 1
sigma <- 1
n <- 100
set.seed(32611)
X <- rnorm(n, mu, sigma)
B <- 1000
Ts <- numeric(B)

for(b in 1:B){
  set.seed(b)
  ax <- sample(X, replace = T)
  Ts[b] <- median(ax)
}

varTest <- var(Ts)
print(varTest)

#' 
#' 
#' Bootstrapping Variance Estimator 
#' ===
#' 
#' 1.  Draw a bootstrapping sample $X_1^*, \ldots, X_n^* \sim F_n$, where $F_n$ is the emperical CDF. Compute ${T^*}_n = g(X_1^*, \ldots, X_n^*)$.
#' 2. Repeat the previous step $B$ times, yielding estimators ${T^*}_n^{(1)}, \ldots, {T^*}_n^{(B)}$.
#' 3. Compute 
#' $$\hat{Var}_{F_n}({T}_n) = \frac{1}{B-1}\sum_{b=1}^B ({T^*}_n^{(b)} - \bar{T}^*)^2,$$
#' where $\bar{T}^* = \frac{1}{B}\sum_{b=1}^B {T^*}_n^{(b)}$
#' 4. Output $\hat{Var}_{F_n}({T}_n)$ as the bootstrapping variance of ${T}_n$.
#' 
#' 
#' 
#' Why Bootstrapping variance works?
#' ===
#' 
#' 
#' - $T_n = g(X_1, \ldots, X_n)$
#' - $mean_F(T_n) = \int g(X_1, \ldots, X_n) f(X) dX = \int g(X_1, \ldots, X_n) dF$
#' - $Var_F(T_n) = \int (g(X_1, \ldots, X_n) - mean_F(T_n))^2 dF$
#' 
#' Since in general, we don't know distribution $F$, we will calculate using the empirical CDF $F_n$.
#' 
#' - $Var_{F_n}(T_n) = \int (g(X_1, \ldots, X_n) - mean_{F_n}(T_n))^2 dF_n$
#' - Finally, we used bootstrapping variance $\hat{Var}_{F_n}(T_n)$ to estimate $Var_{F_n}(T_n)$.
#' 
#' To summarize:
#' 
#' - Estimation error: $Var_F(T_n) - Var_{F_n}(T_n) = O_p(1/\sqrt{n})$
#' - Simulation error: $Var_{F_n}(T_n) - \hat{Var}_{F_n}(T_n) = O_p(1/\sqrt{B})$
#' 
#' 
#' The parametric Bootstrapping (Variance of "median estimator")
#' ===
#' 
#' - Calculate $\hat{\mu} = \arg\max L(\mu;X_1^n)$, $\sigma$ is known.
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n \sim N(\hat{\mu},\sigma^2)$ ($\hat{F}$)
#'     - Compute $\hat{T}^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{\hat{T}}_n = \frac{1}{B} \sum_{b=1}^B \hat{T}^{(b)}_n$
#' - $\hat{V}_F (T_n) =  \frac{1}{B-1} \sum_{b=1}^B \{\hat{T}^{(b)}_n - \bar{\hat{T}}_n \}^2$
#' 
#' 
#' The parametric Bootstrapping (Variance of "median estimator")
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
mu <- 1
sigma <- 1
n <- 100
set.seed(32611)
X <- rnorm(n, mu, sigma)
muhat <- mean(X)
B <- 1000
Ts <- numeric(B)

for(b in 1:B){
  set.seed(b)
  ax <- rnorm(n,muhat,sigma)
  Ts[b] <- median(ax)
}

varTest <- var(Ts)
print(varTest)

#' 
#' 
#' Comparison study
#' ===
#' 
#' - $X_1, \ldots, X_n \sim POI(\lambda)$ ($\lambda = 5$), while we only observe the data $X_1, \ldots, X_n$ and know data is from Poisson.
#' - $n = 100$
#' - $T_n = (\frac{1}{n}\sum_{i=1}^nX_i)^2$
#' - What is the variance of $T_n$?
#' 
#' 
#' - Methods:
#'   - delta method
#'   - non-parametric Bootstrapping
#'   - parametric Bootstrapping
#'   - simulation (requires knowning the underlying parameter $\lambda$)
#' 
#' Delta method
#' ===
#' 
#' - $\hat{\lambda} = \bar{X}$
#' - $var(\bar{X}) = \frac{\lambda}{n}$
#' - $var(T_n) = var(\bar{X}^2) = (2 \lambda)^2 \times var(\bar{X}) = \frac{4\lambda^3}{n}$
#' - $\hat{var}_F(T_n) = \frac{4\hat{\lambda}^3}{n} = \frac{4\bar{X}^3}{n}$
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
set.seed(32611)
X <- rpois(n, lambda)
var_hat1 <- 4*mean(X)^3/n
print(var_hat1)

#' 
#' Bootstrapping (non-parametric)
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
set.seed(32611)
X <- rpois(n, lambda)
B <- 1000
TB <- numeric(B)
for(b in 1:B){
  set.seed(b)
  aX <- sample(X,n,replace = T)
  TB[b] <- (mean(aX))^2
}
var_hat2 <- var(TB)
print(var_hat2)

#' 
#' 
#' Parametric Bootstrapping
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
set.seed(32611)
X <- rpois(n, lambda)
lambdaHat <- mean(X)
B <- 1000
TB <- numeric(B)
for(b in 1:B){
  set.seed(b)
  aX <- rpois(n, lambdaHat)
  TB[b] <- (mean(aX))^2
}
var_hat3 <- var(TB)
print(var_hat3)

#' 
#' 
#' 
#' 
#' Bootstrapping using R package
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
library(boot)
myMean <- function(data, indices){
  d <- data[indices]
  mean(d)^2
}

## non parametric bootstrap
set.seed(32611)
boot_nonpara <- boot(data=X, statistic = myMean, R = B)
var(boot_nonpara$t)

## parametric bootstrap
genPois <- function(data, lambda){
  rpois(length(data), lambda)
}
boot_para <- boot(data=X, statistic = myMean, R = B, sim="parametric", ran.gen = genPois, mle = mean(X))
var(boot_para$t)

#' 
#' 
#' 
#' Simulation
#' ===
#' 
#' - We know the underlying parameters in simulations
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
B <- 1000
Ts <- numeric(B)
for(b in 1:B){
  set.seed(b)
  aX <- rpois(n, lambda)
  Ts[b] <- (mean(aX))^2
}
print(var(Ts))

#' 
#' 
#' Summary
#' ===
#' 
#' - Assume $\sigma$ is known, $\mu$ is the only unknown parameter. 
#' 
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n$ from
#'       - Simulation: $N(\mu,\sigma^2)$
#'       - Non-parametric bootstrapping: $X^{b}_1, \ldots, X^{b}_n \sim \{ X_1, \ldots, X_n \}$ with replacement.
#'       - Parametric bootstrapping: $N(\hat{\mu},\sigma^2)$
#'     - Compute $T^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{T}_n = \frac{1}{B} \sum_{b=1}^B T^{(b)}_n$
#' - $\hat{V}_{F_n} (T_n) =  \frac{1}{B-1} \sum_{b=1}^B \{T^{(b)}_n - \bar{T}_n \}^2$
#' 
#' 
#' 
#' Summary
#' ===
#' 
#' - Goal: to estimate the variance of an estimator
#' 
#' Method      | Need simulation?  | Need underlying parameter  
#' ------------- | ------------- | -------------
#' Delta method    | N |  N
#' Non-parametric bootstrapping   | Y  | N
#' Parametric bootstrapping  | Y   | N
#' Simulation  | Y  | Y
#' 
#' 
#' Comparison between Simulation and Non-parametric bootstrapping
#' ===
#' 
#' - Simulation: Monte Carlo Simulation from underlying distribution.
#' - Non-parametric bootstrapping: Monte Carlo Simulation from the empirical distribution.
#' 
#' The rest of the percedures are the same for these two methods
#' 
#' Bootstrapping confidence interval
#' ===
#' 
#' - Percentiles
#' - normal approximation
#' - Pivotal Intervals
#' 
#' 
#' Bootstrapping confidence interval via Percentiles
#' ===
#' 
#' 1.  Draw a bootstrapping sample $X_1^*, \ldots, X_n^* \sim F_n$. Compute ${T^*}_n = g(X_1^*, \ldots, X_n^*)$.
#' 2. Repeat the previous step $B$ times, yielding estimators ${T^*}_n^{(1)}, \ldots, {T^*}_n^{(B)}$.
#' 3. Rank ${T^*}_n^{(1)}, \ldots, {T^*}_n^{(B)}$ such that
#' ${T^r}_n^{(1)} \le {T^r}_n^{(2)} \le \ldots \le {T^r}_n^{(B)}$
#' 
#' We can define 95\% confidence interval using (if B = 10,000)
#' $$[{T^r}_n^{(250)}, {T^r}_n^{(9750)}]$$
#' 
#' Calculate Bootstrapping confidence interval via Percentiles (1)
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
set.seed(32611)
X <- rpois(n, lambda)
B <- 1000
TB <- numeric(B)
for(b in 1:B){
  set.seed(b)
  aX <- sample(X,n,replace = T)
  TB[b] <- (mean(aX))^2
}
quantile(TB, c(0.025, 0.975))

#' 
#' 
#' Performance of Bootstrapping confidence interval via Percentiles (2)
#' ===
#' 
#' - Underlying truth:
#' 
#' $$E(\bar{X}^2) = E(\bar{X})^2 + var(\bar{X}) = \lambda^2 + \lambda/n$$
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
truth <- lambda^2 + lambda/n
B <- 1000
Repeats <- 100

counts <- 0

plot(c(0,100),c(0,Repeats), type="n", xlab="boot CI", ylab="repeats index")
abline(v = truth, col=2)

for(r in 1:Repeats){
  set.seed(r)
  X <- rpois(n, lambda)
  TB <- numeric(B)
  for(b in 1:B){
    set.seed(b)
    aX <- sample(X,n,replace = T)
    TB[b] <- (mean(aX))^2
  }
  segments(quantile(TB, c(0.025)), r, quantile(TB, c(0.975)), r)
  if(quantile(TB, c(0.025)) < truth & truth < quantile(TB, c(0.975))){
    counts <- counts + 1
  }
}

counts/Repeats

#' 
#' 
#' Calculation of Bootstrapping confidence interval via Percentiles (3)
#' ===
#' 
#' - We can also obtain this Percentiles CI by boot package
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
library(boot)
myMean <- function(data, indices){
  d <- data[indices] ## in this example, data is a vector
  mean(d)^2
}
boot_nonpara <- boot(data=X, statistic = myMean, R = B)
boot.ci(boot_nonpara, type="perc")

#' 
#' 
#' 
#' Normal approximation
#' ===
#' 
#' $$[\hat{T}_n - Z_{1 - \alpha/2}\hat{\sigma}_B, \hat{T}_n - Z_{\alpha/2}\hat{\sigma}_B],$$
#' Where $Z_\alpha = \Phi^{-1}(1-\alpha)$, $\Phi$ is the cdf of standard Normal distribution.
#' 
#' - $Z_{0.025} = -1.96$
#' - $Z_{0.975} = 1.96$
#' 
#' Where $\hat{T}_n$ is the estimator from the original sample and $\hat{\sigma}_B$ is bootstrapping se.
#' 
#' 
#' Implementation for Normal approximation
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
B <- 1000

set.seed(32611)
X <- rpois(n, lambda)
lambdaHat <- mean(X)
That <- lambdaHat^2 
TB <- numeric(B)
for(b in 1:B){
  set.seed(b)
  aX <- sample(X,n,replace = T)
  TB[b] <- (mean(aX))^2
}
ci_l <- That - 1.96*sd(TB)
ci_u <- That + 1.96*sd(TB)

c(ci_l, ci_u)

#' 
#' 
#' Calculation of Bootstrapping confidence interval via Normal approximation (4)
#' ===
#' 
#' - We can also obtain this Percentiles CI by boot package
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
library(boot)
myMean <- function(data, indices){
  d <- data[indices] ## in this example, data is a vector
  mean(d)^2
}
boot_nonpara <- boot(data=X, statistic = myMean, R = B)
boot.ci(boot_nonpara, type="norm")

#' 
#' 
#' Evaluation for Normal approximation
#' ===
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
lambda <- 5
n <- 100
truth <- lambda^2 
B <- 1000
Repeats <- 100

counts <- 0

plot(c(0,100),c(0,Repeats), type="n", xlab="boot CI", ylab="repeats index")
abline(v = truth, col=2)

for(r in 1:Repeats){
  set.seed(r)
  X <- rpois(n, lambda)
  lambdaHat <- mean(X)
  That <- lambdaHat^2 
  TB <- numeric(B)
  for(b in 1:B){
    set.seed(b)
    aX <- sample(X,n,replace = T)
    TB[b] <- (mean(aX))^2
  }
  ci_l <- That - 1.96*sd(TB)
  ci_u <- That + 1.96*sd(TB)
  segments(ci_l, r, ci_u, r)
  if(ci_l < truth & truth < ci_u){
    counts <- counts + 1
  }
}

counts/Repeats

#' 
#' 
#' Bootstrapping confidence interval via Pivotal Intervals 
#' ===
#' 
#' - Won't be covered this year.
#' - If you are interested in this, checkout previous lecture notes <https://caleb-huo.github.io/teaching/2017FALL/lectures/week11_bootstrapAndPermutation/bootstrap/bootstrap.html>
#' 
#' 
#' 
#' Summary Bootstrapping confidence interval
#' ===
#' 
#' Precedure  | Theoritical guarantee | Fast | R package Boot?
#' ------------- | ------------- | ------------- | -------------
#' Percentiles    | No | Yes | Yes
#' Pivotal Intervals       | Yes | No | Yes
#' Pivotal Intervals (simplified, no se) | Yes | Yes | No
#' normal approximation | Yes | Yes | Yes
#' 
#' 
#' Bootstrapping p-value
#' ===
#' 
#' - Using linear regression an example.
#'   - We can obtain the pvalue for lcavol using lm pacakge
#'   - How to get thie pvalue from bootstrapping procedure?
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
library(ElemStatLearn)
prostate$train <- NULL
alm <- lm(lpsa ~ ., data = prostate)
summary(alm)

#' 
#' 
#' Bootstrapping p-value
#' ===
#' 
#' - We are interested in calculating the bootstrapping for lcavol
#' - Estimate the bootstrapping se
#' - Then calculate the Z statistics
#'   $$Z = \frac{\hat{\beta}}{\hat{se(\hat{\beta})}}$$
#' - Under the NULL, $Z\sim N(0,1)$
#'   - We can obtain the pvalue for lcavol using lm pacakge
#'   - How to get thie pvalue from bootstrapping procedure?
#' 
#' 
#' --- 
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
B <- 1000
coef_lcavol <- rep(NA,B)
for(b in 1:B){
  set.seed(b)
  bindex <- sample(nrow(prostate), replace = TRUE)
  bprostate <- prostate[bindex,]
  blm <- lm(lpsa ~ ., data = bprostate)
  coef_lcavol[b] <- coef(blm)["lcavol"]
}
se_lcavol <- sd(coef_lcavol)
Z_lcavol <- coef(alm)["lcavol"] / se_lcavol
pnorm(Z_lcavol, lower.tail = FALSE) * 2 ## two-sided bootstrapping p-value

#' 
#' 
#' Bootstrapping p-value
#' ===
#' 
#' - lasso estimator (when $\lambda = 10$)
#' - this is hard to directly obtain a p-value
#' - But we can use bootstrapping method to get a p-value
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
library(lars)
x <- as.matrix(prostate[,1:8])
y <- prostate[,9]
lassoFit <- lars(x, y, normalize = FALSE) ## lar for least angle regression
coef(lassoFit, s=10, mode="lambda") ## get beta estimate when lambda = 2. 

#' 
#' 
#' 
#' --- 
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
B <- 1000
coef_lcavol <- rep(NA,B)
for(b in 1:B){
  set.seed(b)
  bindex <- sample(nrow(prostate), replace = TRUE)
  bprostate <- prostate[bindex,]
  bx <- as.matrix(bprostate[,1:8])
  by <- bprostate[,9]
  blassoFit <- lars(bx, by, normalize = FALSE) ## lar for least angle regression
  bcoef <- coef(blassoFit, s=10, mode="lambda") ## get beta estimate when lambda = 2. 
  coef_lcavol[b] <- bcoef["lcavol"]
}
se_lcavol <- sd(coef_lcavol)
Z_lcavol <- coef(lassoFit, s=10, mode="lambda")["lcavol"] / se_lcavol
pnorm(Z_lcavol, lower.tail = FALSE) * 2 ## two-sided bootstrapping p-value

#' 
#' 
#' 
#' HW, Large scale Bootstrapping exercise
#' ===
#' 
#' - For the HAPMAP data on hiperGator.
#' - Calculate the sample contrivance matrix.
#' - what is the Bootstrapping variance of the largest eigen value.
#' - what is the Bootstrapping confidence interval of the largest eigen value.
#' 
#' 
#' Permutation test
#' ===
#' 
#' - Distribution free.
#' - Doesn't involve any asymptotic approximations.
#' 
#' 
#' Problem setting:
#' 
#' - Suppose we have data 
#'     - $X_1, \ldots, X_n \sim F$
#'     - $Y_1, \ldots, Y_m \sim G$
#' - We want to test: $H_0: F=G$ versus $H_1: F \ne G$
#' 
#' 
#' Permutation test procedure
#' ===
#' 
#' - If we want to test if the mean parameter of two distributions are the same.
#' - Let $Z = (X_1, \ldots, X_n, Y_1, \ldots, Y_m)$.
#' - Create labels $L = (1,1,\ldots, 1,2,2,\ldots, 2)$ such that there are $n$ copies of 1 and $m$ copies of 2.
#' - A test statistics can be written as a function of $Z$ and $L$. For example,
#' $$T = |\bar{X}_n - \bar{Y}_m|$$
#' or
#' 
#' $$T = |
#' \frac
#' {\sum_{i = 1}^NZ_i I(L_i=1)}
#' {\sum_{i = 1}^N I(L_i=1)}
#' -
#' \frac
#' {\sum_{i = 1}^NZ_i I(L_i=2)}
#' {\sum_{i = 1}^N I(L_i=2)}
#' |,$$
#' where $N = n + m$. Therefore the test statistic $T = g(L,Z)$ is a function of $L$ and $Z$.
#' 
#' 
#' 
#' Permutation test procedure (2)
#' ===
#' 
#' - Define
#' $$p = \frac{1}{N!}\sum_\pi I(g(L_\pi,Z) \ge g(L,Z)),$$
#' 
#' where $L_\pi$ is a permutation of the labels and the sum is over all permutations.
#' 
#' - Under $H_0$, permuting the labels does not change the distribution.
#'     - In other word, $g(L,Z)$ has equal chances of having any rank among all permuted values.
#'     - Under $H_0$, $p \sim Unif(0,1)$
#'     - If we reject when $p<\alpha$, then we have a level $\alpha$ test
#' 
#' 
#' Permutation test procedure (3)
#' ===
#' 
#' Sometimes summing over all possible permutations is infeasible.
#' But it suffices to use a random sample of permutation.
#' 
#' Permutation test procedure:
#' 
#' 1. Calculate the observed test statistic $T = g(L,Z)$
#' 2. Compute a random permutation of the labels $L_k$ and compute $T^{(k)} = g(L_k,Z)$. Do this $K$ times, which will genreate values of $T^{(1)}, \ldots, T^{(K)}$.
#' 3. Compute the p-value as:
#' $$\frac{1}{K} \sum_{k=1}^K I(T^{(k)} \ge T )$$
#' 
#' An illustrating example (no difference in mean)
#' ===
#' 
#' - $n = 100$, $m = 100$
#' - $X_i \sim N(0,1^2)$
#' - $Y_i \sim N(0,2^2)$
#' 
#' If we observe $X_1, \ldots, X_n$ and $Y_1, \ldots, Y_m$, how can we test if the mean of $X$ and mean of $Y$ have a difference.
#' Consider the follow one-sided test:
#' 
#' - $H_0: \mu_x = \mu_y$
#' - $H_A: \mu_x < \mu_y$
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
n <- 100
m <- 100

set.seed(32611)
x <- rnorm(n,0,1)
y <- rnorm(n,0,2)

adataFrame <- data.frame(data=c(x,y),label=gl(2,n))

T <- with(adataFrame, mean(data[label==2] - data[label==1]))

B <- 1000

TB <- numeric(B)

for(b in 1:B){
  set.seed(b)
  bdataFrame <- adataFrame
  bdataFrame$label <- sample(bdataFrame$labe)
  TB[b] <- with(bdataFrame, mean(data[label==2] - data[label==1]))
}

pvalue = mean(TB >= T)
pvalue

#' 
#' 
#' 
#' An illustrating example (with a difference in mean)
#' ===
#' 
#' - $n = 100$, $m = 100$
#' - $X_i \sim N(0,1^2)$
#' - $Y_i \sim N(1,2^2)$
#' 
#' If we observe $X_1, \ldots, X_n$ and $Y_1, \ldots, Y_m$, how can we test if the mean of $X$ and mean of $Y$ have a difference.
#' 
## -----------------------------------------------------------------------------------------------------------------------------------
n <- 100
m <- 100

set.seed(32611)
x <- rnorm(n,0,1)
y <- rnorm(n,0.5,2)

adataFrame <- data.frame(data=c(x,y),label=gl(2,n))

T <- with(adataFrame, mean(data[label==2] - data[label==1]))

B <- 1000

TB <- numeric(B)

for(b in 1:B){
  set.seed(b)
  bdataFrame <- adataFrame
  bdataFrame$label <- sample(bdataFrame$labe)
  TB[b] <- with(bdataFrame, mean(data[label==2] - data[label==1]))
}

mean(TB >= T) 

#' 
#' 
#' 
#' 
#' 
#' Reference
#' ===
#' 
#' - <http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf>
#' - <http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf>
#' - proof of p-value follows $UNIF(0,1)$ under the null
#'   - Under the null, a test statistic $S$ has distribution $F(s)$, p-value $p = Pr(S \le s) = F(s)$
#'   - $Pr(P \le p) = Pr(F^{-1}(P) \le F^{-1}(p)) = Pr(S \le s) = p$
#'   - QED (completion of proof) 
#'   
#'   
#'   
#'   
