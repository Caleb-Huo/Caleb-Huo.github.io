#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "bootsrap and permutation test"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday November 6, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' Outline
#' ===
#' 
#' - bootstrap
#'     - bootstrap variance, confidence interval
#'     - non-parametric bootstrap, parametric bootstrap
#' - permutation test
#'     - size
#'     - power
#' 
#' 
#' Bootstrap
#' ===
#' 
#' - Given $X_1, \ldots, X_n$ samples from a distribution, how to estimate the mean of this distribution? 
#'     - Use $\frac{1}{n} \sum_{i=1}^n X_i$ and weak law of large number.
#' $$\frac{1}{n} \sum_{i=1}^n X_i \rightarrow E(X_i)$$
#' - Given $X_1, \ldots, X_n$ samples from a distribution, how to estimate other parameters (Variance, mean squared)? - Similarly use weak law of large number.
#' 
#' - How to get the standard error of that estimator? How to get a confidence interval?
#'     - Bootstrap
#' 
#' - Although the Bootstrap method is nonparametric, 
#' it can  be used for inference about parameters in both parametric and nonparametric models.
#' 
#' 
#' Bootstrap variance- Aim
#' ===
#' 
#' - Suppose $X_1, \ldots, X_n \sim F$ where $F$ is a distribution.
#' - Let $T_n = g(X_1, \ldots, X_n)$ be a statistic, where $T_n$ is a function of the data.
#' - What is the variance of $T_n$, $V_F(T_n)$?
#' 
#' 
#' A concrete example
#' ===
#' 
#' - Suppose $X_1, \ldots, X_n \sim N(\mu,\sigma^2)$ where $N(\mu,\sigma^2)$ is normal distribution.
#' - Let $T_n = \frac{1}{n} \sum_{i=1}^nX_i$ be a statistic, where $T_n$ is mean of $X_1, \ldots, X_n$.
#' - What is the variance of $T_n$? By weak law of large number,
#' 
#' $$V_F (T_n) = \sigma^2 / n$$
#' 
#' - How to get an estimate of $V_F (T_n) = \sigma^2 / n$?
#' $$\hat{V}_F (T_n) = \hat{\sigma}^2 / n$$
#' 
#' $$\hat{\sigma}^2 = \frac{1}{n - 1}\sum_{i=1}^n (X_i - \bar{X})^2$$
#' 
#' 
#' A example about median
#' ===
#' 
#' - Suppose $X_1, \ldots, X_n \sim N(\mu,\sigma^2)$ where $N(\mu,\sigma^2)$ is normal distribution.
#'     - $X_1, \ldots, X_n$ are known and the distribution parameters are unknown.
#' - Let $T_n = median_{i=1}^n(X_i)$ be a statistic, where $T_n$ is median of $X_1, \ldots, X_n$.
#' - What is the variance of $T_n$? 
#' 
#' Assume $N(\mu,\sigma^2)$ is known
#' 
#' - We can use Monte Carlo simulation method to estimate the variance of $T_n$.
#' 
#' A example about median, Monte Carlo method
#' ===
#' 
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n \sim N(\mu,\sigma^2)$
#'     - Compute $T^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{T}_n = \frac{1}{B} \sum_{b=1}^B T^{(b)}_n$
#' - $\hat{V}_F (T_n) =  \frac{1}{B - 1} \sum_{b=1}^B \{T^{(b)}_n - \bar{T}_n \}^2$
#' - By law of large number, $\hat{V}_F (T_n) \rightarrow Var(T_n)$
#' 
## ------------------------------------------------------------------------
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
#' - What if we don't know $N(\mu,\sigma^2)$?
#' 
#' 
#' Notations
#' ===
#' 
#' - $F = P(X \le x)$ is a distribution function.
#' - We can estimate $F$ with the **empirical distribution function**
#' $\hat{F}_n$, the cdf that puts mass $1/n$ at each data point $X_i$.
#' $$\hat{F}_n (x) = \frac{1}{n} \sum_{i = 1}^n I(X_i \le x)$$
#' where
#' \begin{equation}
#'     I(X_i \le x) =
#'     \begin{cases}
#'       1, & \text{if}\ X_i \le x \\
#'       0, & \text{if}\ X_i > x
#'     \end{cases}
#'   \end{equation}
#' 
#' Empirical process
#' ===
#' 
#' - Glivenko-Cantelli Theorem
#' $$\sup_x | \hat{F}_n(x) - F(x)| \rightarrow 0$$
#'   
#' - Dvoretzky-Kiefer-Wolfowitz inequality, for any $\varepsilon > 0$
#' $$P(\sup_x | \hat{F}_n(x) - F(x)| > \varepsilon) \le 2\exp(-2n\varepsilon^2)$$
#' 
## ------------------------------------------------------------------------
library(ggplot2)
n <- 1000
df <- data.frame(x = c(rnorm(n, 0, 1)))
base <- ggplot(df, aes(x)) + stat_ecdf()
base + stat_function(fun = pnorm, colour = "red") + xlim(c(-3,3))

#' 
#' 
#' A example about median, Bootstrap method 
#' ===
#' 
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n \sim$ ~~N(mu,sigma^2)~~ $F_n$
#'     - Compute $T^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{T}_n = \frac{1}{B} \sum_{b=1}^B T^{(b)}_n$
#' - $\hat{V}_F (T_n) =  \frac{1}{B} \sum_{b=1}^B \{T^{(b)}_n - \bar{T}_n \}^2$
#' 
#' How to sample from the empirical distribution?
#' 
#' - Drawing $X_1^*, \ldots, X_n^*$ from $F_n$ is equivalent to draw $n$ observations, with replacement from the original data $\{X_1, \ldots, X_n\}$.
#' - Therefore, bootstrap sampling is also described as resampling data.
#' 
#' 
#' A example about median, Bootstrap method 
#' ===
#' 
## ------------------------------------------------------------------------
mu <- 1
sigma <- 1
n <- 100
set.seed(32611)
X <- rnorm(n, mu, sigma)
B <- 1000
Ts <- numeric(B)

for(b in 1:B){
  set.seed(b)
  ax <- sample(X, b, replace = T)
  Ts[b] <- median(ax)
}

varTest <- var(Ts)
print(varTest)

#' 
#' 
#' Bootstrap Variance Estimator
#' ===
#' 
#' 1.  Draw a bootstrap sample $X_1^*, \ldots, X_n^* \sim F_n$. Compute $\hat{\theta^*}_n = g(X_1^*, \ldots, X_n^*)$.
#' 2. Repeat the previous step $B$ times, yielding estimators $\hat{\theta^*}_n^{(1)}, \ldots, \hat{\theta^*}_n^{(B)}$.
#' 3. Compute 
#' $$\hat{var}_{F_n}(\hat{\theta}_n) = \frac{1}{B}\sum_{b=1}^B (\hat{\theta^*}_n^{(b)} - \bar{\theta})^2,$$
#' where $\bar{\theta} = \frac{1}{B}\sum_{b=1}^B \hat{\theta^*}_n^{(b)}$
#' 4. Output $\hat{var}_{F_n}(\hat{\theta}_n)$ as the bootstrap variance of $\hat{\theta}_n$.
#' 
#' Why Bootstrap Variance works?
#' ===
#' 
#' Notation:
#' 
#' - $p$ is the probability mass function if $F$ is discrete and probability density function if $F$ is continuous.
#' \begin{equation}
#'     \int g(x) dF(x)=
#'     \begin{cases}
#'       \sum_j g(x_j) p(x_j), & \text{if}\ F\ \text{is discrete} \\
#'       \int g(x)p(x)dx, & \text{if}\ F\ \text{is continuous}
#'     \end{cases}
#'   \end{equation}
#' 
#' 
#' - $\hat{\theta}_n = g(X_1, \ldots, X_n)$
#' - $mean_F(\hat{\theta}_n) = \int g(X_1, \ldots, X_n) dF$
#' - $var_F(\hat{\theta}_n) = \int (g(X_1, \ldots, X_n) - mean_F(\hat{\theta}_n))^2 dF$
#' 
#' Since in general, we don't know distribution $F$, we will calculate using the empirical CDF $F_n$.
#' 
#' - $var_{F_n}(\hat{\theta}_n) = \int (g(X_1, \ldots, X_n) - mean_{F_n}(\hat{\theta}_n))^2 dF_n$
#' - Finally we used bootstrap variance $\hat{var}_{F_n}(\hat{\theta}_n)$ to estimate $var_{F_n}(\hat{\theta}_n)$.
#' 
#' To summarize:
#' 
#' - Estimation error: $var_F(\hat{\theta}_n) - var_{F_n}(\hat{\theta}_n) = O_p(1/\sqrt{n})$
#' - Simulation error: $var_{F_n}(\hat{\theta}_n) - \hat{var}_{F_n}(\hat{\theta}_n) = O_p(1/\sqrt{B})$
#' 
#' 
#' 
#' The parametric Bootstrap (an example for median)
#' ===
#' 
#' - calculate $\hat{\mu} = \arg\min L(\mu;X_1^n)$, $\sigma$ is known.
#' - For $b = 1, \ldots, B$
#'     - draw $X^{b}_1, \ldots, X^{b}_n \sim N(\hat{\mu},\sigma^2)$ (kind of parametric empirical distribution $\hat{F}$)
#'     - Compute $\hat{\theta}^{(b)}_n = median_{i=1}^n(X^{b}_i)$
#' - $\bar{\hat{\theta}}_n = \frac{1}{B} \sum_{b=1}^B \hat{\theta}^{(b)}_n$
#' - $\hat{V}_F (T_n) =  \frac{1}{B} \sum_{b=1}^B \{\hat{\theta}^{(b)}_n - \bar{\hat{\theta}}_n \}^2$
#' 
#' 
## ------------------------------------------------------------------------
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
#' Comparison study using delta method, non-parametric Bootstrap and parametric Bootstrap
#' ===
#' 
#' - $X_1, \ldots, X_n \sim POI(\lambda)$ ($\lambda = 5$), while we only observe the data $X_1, \ldots, X_n$ and know data is from Poisson.
#' - $n = 100$
#' - $T_n = (\frac{1}{n}\sum_{i=1}^nX_i)^2$
#' - What is the variance of $T_n$?
#' 
#' 
#' Delta method
#' ===
#' 
#' - $\hat{\lambda} = \bar{X}$
#' - $var(\bar{X}) = \frac{\lambda}{n}$
#' - $var(T_n) = var(\bar{X}^2) = (2 \lambda)^2 \times var(\bar{X}) = \frac{4\lambda^3}{n}$
#' - $\hat{var}_F(T_n) = \frac{4\hat{\lambda}^3}{n} = \frac{4\bar{X}^3}{n}$
#' 
## ------------------------------------------------------------------------
lambda <- 5
n <- 100
set.seed(32611)
X <- rpois(n, lambda)
var_hat1 <- 4*mean(X)^3/n
print(var_hat1)

#' 
#' Bootstrap (non-parametric Bootstrap)
#' ===
#' 
## ------------------------------------------------------------------------
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
#' Parametric Bootstrap
#' ===
#' 
## ------------------------------------------------------------------------
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
#' Bootstrap using R package
#' ===
#' 
## ------------------------------------------------------------------------
library(boot)
myMean <- function(data, indices){
  d <- data[indices]
  mean(d)^2
}

## non parametric bootstrap
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
#' Bootstrap confidence interval
#' ===
#' 
#' - Percentiles
#' - Pivotal Intervals
#' - Pivotal Intervals (simplified)
#' - normal approximation
#' 
#' 
#' Bootstrap confidence interval via Percentiles (Empirically performs well, but there is no theoretical guarantee to work)
#' ===
#' 
#' 1.  Draw a bootstrap sample $X_1^*, \ldots, X_n^* \sim F_n$. Compute $\hat{\theta^*}_n = g(X_1^*, \ldots, X_n^*)$.
#' 2. Repeat the previous step $B$ times, yielding estimators $\hat{\theta^*}_n^{(1)}, \ldots, \hat{\theta^*}_n^{(B)}$.
#' 3. Rank $\hat{\theta^*}_n^{(1)}, \ldots, \hat{\theta^*}_n^{(B)}$ such that
#' $\hat{\theta^r}_n^{(1)} \le \hat{\theta^r}_n^{(2)} \le \ldots \le \hat{\theta^r}_n^{(B)}$
#' 
#' We can define 95\% confidence interval using (if B = 10,000)
#' $$[\hat{\theta^r}_n^{(250)}, \hat{\theta^r}_n^{(9750)}]$$
#' 
#' Performance of Bootstrap confidence interval via Percentiles (1)
#' ===
#' 
## ------------------------------------------------------------------------
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
#' Performance of Bootstrap confidence interval via Percentiles (2)
#' ===
#' 
#' - Underlying truth:
#' 
#' $$E(\bar{X}^2) = E(\bar{X})^2 + var(\bar{X}) = \lambda^2 + \lambda/n$$
#' 
## ------------------------------------------------------------------------
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
#' Performance of Bootstrap confidence interval via Percentiles (3)
#' ===
#' 
#' - We can also obtain this Percentiles CI by boot package
#' 
## ------------------------------------------------------------------------
library(boot)
myMean <- function(data, indices){
  d <- data[indices]
  mean(d)^2
}
boot_nonpara <- boot(data=X, statistic = myMean, R = B)
boot.ci(boot_nonpara, type="perc")
?boot.ci

#' 
#' 
#' Bootstrap confidence interval via Pivotal Intervals (better accuracy and theoretical guarantee), (There is a improved way to do this later)
#' ===
#' 
#' 1.  Draw a bootstrap sample $X_1^*, \ldots, X_n^* \sim F_n$. Compute $\hat{\theta^*}_n = g(X_1^*, \ldots, X_n^*)$.
#' 2. Repeat the previous step $B$ times, yielding 
#'     - estimators $\hat{\theta^*}_n^{(1)}, \ldots, \hat{\theta^*}_n^{(B)}$
#'     - standard error $\hat{\sigma^*}_n^{(1)}, \ldots, \hat{\sigma^*}_n^{(B)}$
#' 3. Compute
#'     - $t_b^* = \frac{\hat{\theta^*}_n^{(b)} - \hat{\theta}_n}{\hat{\sigma^*}_n^{(b)}}$
#' 4. Given $t_b^*$ for $b \in \{1, \ldots, B\}$, define the $\alpha$ quantile $q_\alpha$ as 
#' $$\sum_{b=1}^B I(t_b^* \le q_\alpha)/B = \alpha$$
#' \begin{align}
#' 1 - \alpha & = P(q_{\alpha/2} < t_b^* < q_{1 - \alpha/2}) \\
#' & \approx P(q_{\alpha/2} < t < q_{1 - \alpha/2}) \\
#' & = P(q_{\alpha/2}\hat{\sigma}_B < \hat{\theta}_n - \theta < q_{1 - \alpha/2}\hat{\sigma}_B) \\
#' & = P( \hat{\theta}_n -  q_{1 - \alpha/2}\hat{\sigma}_B < \theta < \hat{\theta}_n - q_{\alpha/2}\hat{\sigma}_B ) 
#' \end{align}
#' 
#' We can define $1 - \alpha$ confidence interval using
#' $$[\hat{\theta}_n - q_{1 - \alpha/2}\hat{\sigma}_B, \hat{\theta}_n - q_{\alpha/2}\hat{\sigma}_B],$$
#' 
#' Where $\hat{\theta}_n$ is the estimator from the original sample and $\hat{\sigma}_B$ is bootstrap se.
#' 
#' If $\hat{\sigma^*}_n^{(b)}$ is hard to estiamte, 
#' we need to use another round of bootstrap to estiamte.
#' 
#' 
#' Performance of Bootstrap confidence interval via Pivotal Intervals (1)
#' ===
#' 
## ------------------------------------------------------------------------
lambda <- 5
n <- 100
set.seed(32611)
X <- rpois(n, lambda)
B <- 1000
TB <- numeric(B)
seB <- numeric(B)
tB <- numeric(B)
lambdaHat <- mean(X)
That <- lambdaHat^2 + lambdaHat/n
for(b in 1:B){
  set.seed(b)
  aX <- sample(X,n,replace = T)
  TB[b] <- (mean(aX))^2
  seB[b] <- sqrt(4*lambdaHat^3/n^2)
  tB[b] <- (TB[b] - That)/seB[b]
}

se_boot <- sd(TB)/sqrt(n)

CI_l <- That - quantile(tB, 0.975) * se_boot
CI_h <- That - quantile(tB, 0.025) * se_boot

print(c(CI_l, CI_h))

#' 
#' 
#' Performance of Bootstrap confidence interval via Pivotal Intervals (2)
#' ===
#' 
#' 
## ------------------------------------------------------------------------
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
  seB <- numeric(B)
  tB <- numeric(B)
  lambdaHat <- mean(X)
  That <- lambdaHat^2 + lambdaHat/n
  for(b in 1:B){
    set.seed(b)
    aX <- sample(X,n,replace = T)
    TB[b] <- (mean(aX))^2
    seB[b] <- sqrt(4*lambdaHat^3/n^2)
    tB[b] <- (TB[b] - That)/seB[b]
  }
  se_boot <- sd(TB)/sqrt(n)

  CI_l <- That - quantile(tB, 0.975) * se_boot
  CI_h <- That - quantile(tB, 0.025) * se_boot
  segments(CI_l, r, CI_h, r)
  if(CI_l < truth & truth < CI_h){
    counts <- counts + 1
  }
}

counts/Repeats

#' 
#' 
#' Bootstrap confidence interval via Pivotal Intervals - no SE (better accuracy and theoretical guarantee)
#' ===
#' 
#' 1.  Draw a bootstrap sample $X_1^*, \ldots, X_n^* \sim F_n$. Compute $\hat{\theta^*}_n = g(X_1^*, \ldots, X_n^*)$.
#' 2. Repeat the previous step $B$ times, yielding 
#'     - estimators $\hat{\theta^*}_n^{(1)}, \ldots, \hat{\theta^*}_n^{(B)}$
#' 3. Compute
#'     - $t_b^* = \hat{\theta^*}_n^{(b)} - \hat{\theta}_n$
#' 4. Given $t_b^*$ for $b \in \{1, \ldots, B\}$, define the $\alpha$ quantile $q_\alpha$ as 
#' $$\sum_{b=1}^B I(t_b^* \le q_\alpha)/B = \alpha$$
#' \begin{align}
#' 1 - \alpha & = P(q_{\alpha/2} < t_b^* < q_{1 - \alpha/2}) \\
#' & \approx P(q_{\alpha/2} < t < q_{1 - \alpha/2}) \\
#' & = P(q_{\alpha/2} < \hat{\theta}_n - \theta < q_{1 - \alpha/2}) \\
#' & = P( \hat{\theta}_n -  q_{1 - \alpha/2} < \theta < \hat{\theta}_n - q_{\alpha/2} ) 
#' \end{align}
#' 
#' We can define $1 - \alpha$ confidence interval using
#' $$[\hat{\theta}_n - q_{1 - \alpha/2}\hat{\sigma}_B, \hat{\theta}_n - q_{\alpha/2}\hat{\sigma}_B],$$
#' 
#' Where $\hat{\theta}_n$ is the estimator from the original sample and $\hat{\sigma}_B$ is bootstrap se.
#' 
#' 
#' 
#' Performance of Bootstrap confidence interval via Pivotal Intervals (no SE)
#' ===
#' 
#' - Remove SE term.
#' 
## ------------------------------------------------------------------------
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
  seB <- numeric(B)
  tB <- numeric(B)
  lambdaHat <- mean(X)
  That <- lambdaHat^2 + lambdaHat/n
  for(b in 1:B){
    set.seed(b)
    aX <- sample(X,n,replace = T)
    TB[b] <- (mean(aX))^2
    tB[b] <- (TB[b] - That)
  }

  CI_l <- That - quantile(tB, 0.975) 
  CI_h <- That - quantile(tB, 0.025) 
  segments(CI_l, r, CI_h, r)
  if(CI_l < truth & truth < CI_h){
    counts <- counts + 1
  }
}

counts/Repeats

#' 
#' 
#' Normal approximation
#' ===
#' 
#' $$[\hat{\theta}_n - Z_{1 - \alpha/2}\hat{\sigma}_B, \hat{\theta}_n - Z_{\alpha/2}\hat{\sigma}_B],$$
#' Where $Z_\alpha = \Phi^{-1}(1-\alpha)$, $\Phi$ is the cdf of standard Normal distribution.
#' 
#' - $Z_{0.025} = -1.96$
#' - $Z_{0.975} = 1.96$
#' 
#' Where $\hat{\theta}_n$ is the estimator from the original sample and $\hat{\sigma}_B$ is bootstrap se.
#' 
#' 
#' experiment for Normal approximation
#' ===
#' 
## ------------------------------------------------------------------------
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
  lambdaHat <- mean(X)
  That <- lambdaHat^2 + lambdaHat/n
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
#' Summary Bootstrap confidence interval
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
#' Large scale bootstrap exercise
#' ===
#' 
#' - For the HAPMAP data Chr1.rdata.
#' - Calculate the sample contrivance matrix.
#' - what is the bootstrap variance of the largest eigen value.
#' - what is the bootstrap confidence interval of the largest eigen value.
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
#' $$p = \frac{1}{N!}\sum_\pi I(g(L_\pi,Z) > g(L,Z)),$$
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
#' 2. Compute a random permutation of the labels $L_k$ and compute $T^{(k)} = g(L_k,Z)$. Do this $K$ times giving values of $T^{(1)}, \ldots, T^{(K)}$.
#' 3. Compute the p-value as:
#' $$\frac{1}{K} \sum_{k=1}^K I(T^{(k)} \ge T )$$
#' 
#' An illustrating example
#' ===
#' 
#' - $n = 100$, $m = 100$
#' - $X_i \sim N(0,1^2)$
#' - $Y_i \sim N(0,2^2)$
#' 
#' If we observe $X_1, \ldots, X_n$ and $Y_1, \ldots, Y_m$, how can we test if the mean of $X$ and mean of $Y$ have a difference.
#' 
## ------------------------------------------------------------------------
n <- 100
m <- 100

set.seed(32611)
x <- rnorm(n,0,1)
y <- rnorm(n,0,2)

adataFrame <- data.frame(data=c(x,y),label=gl(2,n))

T <- with(adataFrame, mean(data[label==1] - data[label==2]))

B <- 1000

TB <- numeric(B)

for(b in 1:B){
  set.seed(b)
  bdataFrame <- adataFrame
  bdataFrame$label <- sample(bdataFrame$labe)
  TB[b] <- with(bdataFrame, mean(data[label==1] - data[label==2]))
}

sum(TB >= T)/B

#' 
#' 
#' Examine the size of the test
#' ===
#' 
#' 
## ------------------------------------------------------------------------
n <- 100
m <- 100
R <- 300
B <- 300
alpha <- 0.05

counts <- 0
for(r in 1:R){
  set.seed(r)
  x <- rnorm(n,0,1)
  y <- rnorm(n,0,2)
  
  adataFrame <- data.frame(data=c(x,y),label=gl(2,n))
  
  T <- with(adataFrame, mean(data[label==1] - data[label==2]))
  
  
  TB <- numeric(B)
  
  for(b in 1:B){
    set.seed(b*r)
    bdataFrame <- adataFrame
    bdataFrame$label <- sample(bdataFrame$labe)
    TB[b] <- with(bdataFrame, mean(data[label==1] - data[label==2]))
  }
  
  if(sum(TB >= T)/B > alpha){
    counts <- counts + 1
  }
}
counts/B

#' 
#' Compare power of permutation test and t.test
#' ===
#' 
#' Consider $H_0: \mu_X = \mu_Y$ versus $H_A: \mu_X > \mu_Y$
#' 
#' - $n = 100$, $m = 100$
#' - $X_i \sim N(1,3^2)$
#' - $Y_i \sim N(0,3^2)$
#' 
#' 
## ------------------------------------------------------------------------
n <- 100
m <- 100
R <- 300
B <- 300
alpha <- 0.05

counts_ttest <- 0
counts_perm <- 0

for(r in 1:R){
  set.seed(r)
  x <- rnorm(n,1,3)
  y <- rnorm(n,0,3)
  
  if(t.test(x,y,alternative = "greater")$p.val < alpha){
    counts_ttest <- counts_ttest + 1 
  }

  adataFrame <- data.frame(data=c(x,y),label=gl(2,n))
  
  T <- with(adataFrame, mean(data[label==1] - data[label==2]))
  TB <- numeric(B)
  
  for(b in 1:B){
    set.seed(b*r)
    bdataFrame <- adataFrame
    bdataFrame$label <- sample(bdataFrame$labe)
    TB[b] <- with(bdataFrame, mean(data[label==1] - data[label==2]))
  }
  
  if(sum(TB >= T)/B < alpha){
    counts_perm <- counts_perm + 1
  }
}
counts_ttest/B
counts_perm/B

#' 
#' 
#' When to use permutation test
#' ===
#' 
#' - If the underlying distribution is unknown or can be assumed, use parametric test
#'     - Quick
#'     - Powerful
#' - If the underlying distribution is unknown or it is very hard to derive parametric distribution of the test statistics, use permutation test
#'     - Correct
#'     - Slow
#'     - May not be as powerful as parametric test (depend on the underlying distribution of the data)
#' 
#' 
#' 
#' Reference
#' ===
#' 
#' - <http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf>
#' - <http://users.stat.umn.edu/~helwig/notes/bootci-Notes.pdf>
#' - <http://www.stat.cmu.edu/~larry/=stat705/Lecture10.pdf>
#' 
#' 
## ------------------------------------------------------------------------
knitr::purl("bootstrap.rmd", output = "bootstrap.R ", documentation = 2)

