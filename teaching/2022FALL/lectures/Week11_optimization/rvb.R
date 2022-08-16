#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Simulating random variables"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday November 3, 2021"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' 
#' Outlines
#' ===
#' 
#' - Simulating random variables
#'   - Direct simulation using R 
#'   - Inverse CDF
#' 
#' 
#' Random samples from a given pool of numbers
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
a <- 1:10
sample(x = a, size = 2)

sample(x = a, size = 2)

sample(x = a, size = 2)

#' 
#' 
#' Are they really random? Can generate the same random number (for reproducibility)?
#' ===
#' 
#' - The random numbers generated in R are not truly random, they are **pseudorandom**.
#' - So we are able to reproduce the exact same random numbers by setting random seeds.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
a <- 1:10
set.seed(32611) ## 32611 can be replaced by any number
sample(x = a, size = 2)
set.seed(32611) ## if you keep the same random seed, you will end up with the exact same result
sample(x = a, size = 2)
set.seed(32611) ## if you keep the same random seed, you will end up with the exact same result
sample(x = a, size = 2)

#' 
#' 
#' Each time the seed is set, the same sequence follows
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
set.seed(32611)
sample(1:10,2); sample(1:10,2); sample(1:10,2)

set.seed(32611)
sample(1:10,2); sample(1:10,2); sample(1:10,2)

#' 
#' However, R version may impact the random number
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
sessionInfo()
set.seed(32611)
sample(1:10,2)

#' 
#' 
#' Want to sample from a given pool of numbers with replacement
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
sample(1:10) ## default is without replacement

#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
sample(1:10, replace = T) ## with replacement

#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
sample(LETTERS[1:10], replace = T) ## with replacement

#' 
#' Random number generation from normal distribution
#' ===
#' 
#' For normal distribution:
#' 
#' - rnorm(): generate random variable from normal distribution
#' - pnorm(): get CDF/pvalue from normal distribution, $\Phi(x) = P(Z \le x)$
#' - dnorm(): normal density function $\phi = \Phi'(x)$
#' - qnorm(): quantile function for normal distribution $q(y) = \Phi^{-1}(y)$ such that $\Phi(q(y)) = y$
#' 
#' ![](../figure/normalPlot.png)
#' 
#' 
#' Random variable simulators in R
#' ===
#' 
#' Distribution  | R command
#' ------------- | -------------
#' binomial      | rbinom
#' Poisson       | rpois
#' geometric     | rgeom
#' negative binomial | rnbinom
#' uniform       | runif
#' exponential   | rexp
#' normal        | rnorm
#' gamma         | rgamma
#' beta          | rbeta
#' student t     | rt
#' F             | rf
#' chi-squared   | rchisq
#' Weibull       | rweibull
#' log normal    | rlnorm
#' 
#' Random variable density in R
#' ===
#' 
#' Distribution  | R command
#' ------------- | -------------
#' binomial      | dbinom
#' Poisson       | dpois
#' geometric     | dgeom
#' negative binomial | dnbinom
#' uniform       | dunif
#' exponential   | dexp
#' normal        | dnorm
#' gamma         | dgamma
#' beta          | dbeta
#' student t     | dt
#' F             | df
#' chi-squared   | dchisq
#' Weibull       | dweibull
#' log normal    | dlnorm
#' 
#' 
#' Random variable distribution tail probablity in R
#' ===
#' 
#' Distribution  | R command
#' ------------- | -------------
#' binomial      | pbinom
#' Poisson       | ppois
#' geometric     | pgeom
#' negative binomial | pnbinom
#' uniform       | punif
#' exponential   | pexp
#' normal        | pnorm
#' gamma         | pgamma
#' beta          | pbeta
#' student t     | pt
#' F             | pf
#' chi-squared   | pchisq
#' Weibull       | pweibull
#' log normal    | plnorm
#' 
#' 
#' Random variable distribution quantile in R
#' ===
#' 
#' Distribution  | R command
#' ------------- | -------------
#' binomial      | qbinom
#' Poisson       | qpois
#' geometric     | qgeom
#' negative binomial | qnbinom
#' uniform       | qunif
#' exponential   | qexp
#' normal        | qnorm
#' gamma         | qgamma
#' beta          | qbeta
#' student t     | qt
#' F             | qf
#' chi-squared   | qchisq
#' Weibull       | qweibull
#' log normal    | qlnorm
#' 
#' 
#' Normal distribution
#' ===
#' 
#' $$f(x;\mu,\sigma) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
aseq <- seq(-4,4,.01)
plot(aseq,dnorm(aseq, 0, 1),type='l', xlab='x', ylab='Density', lwd=2)
lines(aseq,dnorm(aseq, 1, 1),col=2, lwd=2)
lines(aseq,dnorm(aseq,0, 2),col=3, lwd=2)
legend("topleft",c(expression(paste(mu==0, ", " ,  sigma==1 ,sep=' ')), 
             expression(paste(mu==1, ", " ,  sigma==1 ,sep=' ')), 
             expression(paste(mu==0, ", " ,  sigma==2 ,sep=' '))), 
       col=1:3, lty=c(1,1,1), lwd=c(2,2,2), cex=1, bty='n')
mtext(side=3,line=.5,'Normal distributions',cex=1, font=2)

#' 
#' t distribution
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
aseq <- seq(-4,4,.01)
plot(aseq,dnorm(aseq),type='l', xlab='x', ylab='Density', lwd=2) ## same as dt(aseq, Inf)
lines(aseq,dt(aseq,10),col=2, lwd=2)
lines(aseq,dt(aseq,4),col=3, lwd=2)
lines(aseq,dt(aseq,2),col=4, lwd=2)
legend("topleft",c(expression(normal), expression(paste(df==10,sep=' ')), 
             expression(paste(df==4,sep=' ')), 
             expression(paste(df==2,sep=' '))), 
       col=1:4, lty=c(1,1,1), lwd=c(2,2,2), cex=1, bty='n')
mtext(side=3,line=.5,'t distributions',cex=1, font=2)

#' 
#' 
#' Chi-square distribution
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
aseq <- seq(0,4,.01)
plot(aseq,dchisq(aseq, df = 1),type='l', xlab='x', ylab='Density', lwd=2, ylim = c(0,1))
lines(aseq,dchisq(aseq,df = 2),col=2, lwd=2)
lines(aseq,dchisq(aseq,df = 5),col=3, lwd=2)
lines(aseq,dchisq(aseq,df = 10),col=4, lwd=2)
legend("topright",c(expression({Chi^2}[(1)]), expression({Chi^2}[(2)]), 
             expression({Chi^2}[(5)]), 
             expression({Chi^2}[(10)])
             ), 
       col=1:4, lty=c(1,1,1), lwd=c(2,2,2), cex=1, bty='n')
mtext(side=3,line=.5,'Chi square distributions',cex=1, font=2)

#' 
#' 
#' Chi-square distribution
#' ===
#' 
#' - $x \sim \chi^2(k)$
#' 
#' - $E(x) = 2k$
#' 
#' - If $x \sim N(0,1)$, then $x^2 \sim \chi^2(1)$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
## verification via qq-plot
set.seed(32608)
n <- 1000
x1 <- rnorm(n)
x2 <- rchisq(n,df = 1)
## the best practice is to use the theoretical distribution
## for simplicity, we can also use the emperical distribution

s1_sorted <- sort(x1^2)
s2_sorted <- sort(x2)

plot(s1_sorted, s2_sorted, xlim = c(0,5), ylim = c(0,5))

#' 
#' 
#' Poisson distribution
#' ===
#' 
#' $$f(k;\lambda) = \frac{\lambda^k e^{-\lambda}}{k!},$$
#' where $k$ is non negative integer.
## ----------------------------------------------------------------------------------------------------------------------------------------------
aseq <- seq(0,8,1)
par(mfrow=c(2,2))
for(i in 1:4){
  counts <- dpois(aseq,i)
  names(counts) <- aseq
  barplot(counts, xlab='x', ylab='Density', lwd=2, col=i, las = 2,  main=bquote(paste(lambda==.(i), sep=' ')), ylim=c(0, 0.4))
}



#' 
#' 
#' Beta distribution
#' ===
#' 
#' $$f(x;\alpha,\beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1}(1 - x)^{\beta - 1}$$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
aseq <- seq(.001,.999,.001)
plot(aseq,dbeta(aseq,.25,.25), type='l', ylim=c(0,6), ylab='Density', xlab='Proportion (p)',lwd=2)
lines(aseq, dbeta(aseq,2,2),lty=2,lwd=2)
lines(aseq, dbeta(aseq,2,5),lty=1,col=2,lwd=2)
lines(aseq, dbeta(aseq,12,2),lty=2,col=2,lwd=2)
lines(aseq, dbeta(aseq,20,.75),lty=1,col='green',lwd=2)
lines(aseq, dbeta(aseq,1,1),lty=2,lwd=2, col=4)
legend(.2,6,c(expression(paste(alpha==.25,', ', beta==.25)), expression(paste(alpha==2,', ',beta==2)), expression(paste(alpha==2,', ', beta==5)), expression(paste(alpha==12,', ',beta==2)), expression(paste(alpha==20,', ',beta==.75)), expression(paste(alpha==1,', ', beta==1))), lty=c(1,2,1,2,1,2), col=c(1,1,2,2,'green',4), cex=1,bty='n',lwd=rep(2,6))
mtext(side=3,line=.5,'Beta distributions',cex=1, font=2)

#' 
#' Beta distribution
#' ===
#' 
#' - $x \sim \mbox{Beta}(\alpha, \beta)$
#' 
#' - $f(x;\alpha,\beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1}(1 - x)^{\beta - 1}$
#' 
#' - $E(x) = \frac{\alpha}{\alpha + \beta}$
#' 
#' - when $\alpha = \beta = 1$, Beta distribution reduces to UNIF(0, 1)
#' 
#' Gamma distribution
#' ===
#' 
#' $$f(x;k,\theta) = \frac{1}{\Gamma(k) \theta^k} x^{k-1}e^{-\frac{x}{\theta}}$$
#' 
#' - $k$: shape parameter
#' - $\theta$: scale parameter
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
aseq <- seq(0,7,.01)
plot(aseq,dgamma(aseq,shape=1,scale=1),type='l', xlab='x', ylab='Density', lwd=2)
lines(aseq,dgamma(aseq,shape=2,scale=1),col=4, lwd=2)
lines(aseq,dgamma(aseq,shape=4,scale=4),col=2, lwd=2)
legend(3,1,c(expression(paste(k==1,', ',theta==1,sep=' ')), expression(paste(k==2,', ',theta==1,sep=' ')), expression(paste(k==4,', ', theta==4,sep=' '))), col=c(1,4,2), lty=c(1,1,1), lwd=c(2,2,2), cex=1, bty='n')
mtext(side=3,line=.5,'Gamma distributions',cex=1, font=2)

#' 
#' Gamma distribution
#' ===
#' 
#' - $x \sim \mbox{Gamma}(k, \theta)$
#' 
#' - $f(x;k,\theta) = \frac{1}{\Gamma(k) \theta^k} x^{k-1}e^{-\frac{x}{\theta}}$
#' 
#' - $y \sim \mbox{EXP}(\theta)$ if $y \sim \mbox{Gamma}(1, \theta)$
#' 
#' - $y \sim \chi^2(v)$ if $y \sim \mbox{Gamma}(v/2, 2)$
#' 
#' 
#' 
#' 
#' log normal distribution
#' ===
#' 
#' A positive random variable $X$ is log-normally distributed if the logarithm of X is normally distributed,
#' $$\ln(X) \sim N(\mu, \sigma^2)$$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
aseq <- seq(0,7,.01)
plot(aseq,dlnorm(aseq,.1,2),type='l', xlab='x', ylab='Density', lwd=2)
lines(aseq,dlnorm(aseq,2,1),col=4, lwd=2)
lines(aseq,dlnorm(aseq,0,1),col=2, lwd=2)
legend(3,1.2,c(expression(paste(mu==0.1,', ',sigma==2,sep=' ')), expression(paste(mu==2,', ',sigma==1,sep=' ')), expression(paste(mu==0,', ',sigma==1,sep=' '))), col=c(1,4,2), lty=c(1,1,1), lwd=c(2,2,2), cex=1,bty='n')
mtext(side=3,line=.5,'Lognormal distributions',cex=1, font=2)

#' 
#' 
#' samples from Normal distribution
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
set.seed(32608)
z <- rnorm(1000)
hist(z,nclass=50, freq = F)
lines(density(z), col=2, lwd=2)
curve(dnorm, from = -3, to = 4 ,col=4, lwd=2, add = T)
legend("topright", c("empirical", "theoritical"), col=c(2,4), lwd=c(2,2))

#' 
#' Check CDF
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
set.seed(32608)
z <- rnorm(1000)
plot(ecdf(z), ylab="Distribution", main="Empirical distribution",
     lwd=2, col="red")
curve(pnorm, from = -3, to = 3, lwd=2, add = T)
legend("topleft", legend=c("Empirical distribution", "Actual distribution"),
       lwd=2, col=c("red","black"))

#' 
#' 
#' In class exercise: Verify weak law of large number
#' ===
#' 
#' The **weak law of large number** states the sample average converges in probability towards the expected value.
#' 
#' $$\frac{1}{n} \sum_{i=1}^n X_i \rightarrow \mathbb{E}(X)$$
#' 
#' - Suppose $X$ are generated from exponential distribution with mean parameter 1. Verify weak law of large number by simulation.
#'     - set $n = 10, 100, 1000, \ldots, 10^8$
#'     - Simulate n samples from $EXP(1)$
#'     - Calculate the mean value of these n random samples, repeat for different n.
#'     - plot scattered plot of y: abs(mean value - expectation) vs x: n (y should be in log scale).
#' 
## ---- eval = FALSE-----------------------------------------------------------------------------------------------------------------------------
## n <- 10^seq(2,8)
## errors <- numeric(length(n))
## 
## for(i in seq_along(n)){
##   set.seed(i)
##   an <- n[i]
##   ax <- rexp(an)
##   errors[i] <- mean(ax) - 1
## }
## 
## plot(n, errors, log = "xy")

#' 
#' 
#' p-value 
#' ===
#' 
#' - p-value:
#'     - When null hypothesis is true, the probability that the null statistics is as extreme or more extreme than the observed statistics.
#' 
#' ![](../figure/pvalue.jpeg)
#' 
#' - p-value is the cumulative density, can be computed by pnorm() function
#' 
#' p-value example
#' ===
#' 
#' - E.g Null is $N(0,1)$, an observed statistics is 2.
#'     - one sided test, p-value is 
## ----------------------------------------------------------------------------------------------------------------------------------------------
pnorm(q = 2, mean = 0, sd = 1, lower.tail = FALSE) ## default: lower.tail = TRUE

#' 
#'     - two sided test, p-value is 
#'     
## ----------------------------------------------------------------------------------------------------------------------------------------------
pnorm(q = 2, mean = 0, sd = 1, lower.tail = FALSE) + pnorm(q = - 2, mean = 0, sd = 1, lower.tail = TRUE)

#' 
#' quantile: qnorm()
#' ===
#' 
#' - qnorm() calculates: given the area under the density curve (cumulative density), what is the quantile to generate such cumulative density?
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
qnorm(p = 0.975, mean = 0, sd = 1, lower.tail = TRUE) 
qnorm(p = 0.025, mean = 0, sd = 1, lower.tail = TRUE) 

#' 
#' - this is why `r signif(qnorm(p = 0.975, mean = 0, sd = 1, lower.tail = TRUE), 3)` corresponds to 95% confidence interval.
#' 
#' ![](../figure/ci95.gif)
#' 
#' Multi-variate normal distribution
#' ===
#' 
#' $$X \sim N(
#' \begin{pmatrix}
#' \mu_1 \\ 
#' \mu_2 
#' \end{pmatrix}, 
#' \begin{pmatrix}
#' \sigma_{11} & \sigma_{12} \\ 
#' \sigma_{21} & \sigma_{22}
#' \end{pmatrix}
#' )$$
#' 
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
mu <- c(0,0)
Sigma <- matrix(c(10,3,3,2),2,2)
Sigma
var(MASS::mvrnorm(n = 1000, mu, Sigma))

#' 
#' Multi-variate normal distribution simulation
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
mu <- c(0,0)
Sigma1 <- matrix(c(1,0,0,1),2,2)
Sigma2 <- matrix(c(1,0,0,5),2,2)
Sigma3 <- matrix(c(5,0,0,1),2,2)
Sigma4 <- matrix(c(1,0.8,0.8,1),2,2)
Sigma5 <- matrix(c(1,1,1,1),2,2)
Sigma6 <- matrix(c(1,-0.8,-0.8,1),2,2)
arange <- c(-6,6)
par(mfrow=c(2,3))
plot(MASS::mvrnorm(n = 1000, mu, Sigma1), xlim=arange, ylim=arange)
plot(MASS::mvrnorm(n = 1000, mu, Sigma2), xlim=arange, ylim=arange)
plot(MASS::mvrnorm(n = 1000, mu, Sigma3), xlim=arange, ylim=arange)
plot(MASS::mvrnorm(n = 1000, mu, Sigma4), xlim=arange, ylim=arange)
plot(MASS::mvrnorm(n = 1000, mu, Sigma5), xlim=arange, ylim=arange)
plot(MASS::mvrnorm(n = 1000, mu, Sigma6), xlim=arange, ylim=arange)

#' 
#' 
#' Truncated distribution
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
library(truncdist)

# dtrunc
# rtrunc
# ptrunc
# qtrunc

#' 
#' generate samples from truncated normal distribution
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
x <- rtrunc( 1000, spec="norm", a=1, b=Inf )
hist(x, nclass = 50)

#' 
#' density of truncated normal distribution
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
truncf <- function(x) dtrunc(x, spec="norm", a=1, b=Inf )
curve(truncf, from = -3, to = 3)

#' 
#' What kinds of distribution (spec) does this function support
#' ===
#' 
#' - Seems to be all distributions available in r* (e.g. rnorm).
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
dtrunc(0.5, spec="norm", a=-2, b=Inf )
dtrunc(0.5, spec="beta", shape1=2, shape2 = 2, a=0.3, b=Inf )
dtrunc(0.5, spec="gamma", shape=2, rate = 2, a=0.3, b=Inf )

#' 
#' 
#' Another way generate random samples
#' ===
#' 
#' - In the era without R, how people generate random samples from a distribution?
#' - Inverse CDF transformation
#' 
#' 
#' CDF of any distribution function follows UNIF(0,1)
#' ===
#' 
#' - $X$ is random variable from certain distribution.
#' - $F_X$ is CDF of $X$, assume to be continuous and increasing.
#' - Define $Z = F_X(X)$ and range of $Z$ is $[0, 1]$.
#' 
#' $$F_Z(y) = P(F_X(X) \le y) = P(X \le F_X^{-1}(y)) = F_X(F_X^{-1}(y)) = y$$
#' - If $U$ is a uniform random variable who takes values in $[0, 1]$,
#' $$F_U(y) = \int_{-\infty}^y f_U(u) du = \int_0^y du = y$$
#' 
#' Thus $Z \sim UNIF(0, 1)$
#' 
#' - Therefore, instead of directly sampling from $X$, we can sample from $Z \sim UNIF(0, 1)$ and calculate $F_X^{-1}(Z)$.
#' 
#' Beta distribution 
#' ===
#' 
#' - $x \sim \mbox{Beta}(\alpha, \beta)$
#' 
#' - $f(x;\alpha,\beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1}(1 - x)^{\beta - 1}$
#' 
#' - Beta(3,1)
#' - PDF: $f(x) = 3x^2$
#' - CDF: $F(x) = x^3$
#' - Inverse CDF: $F^{-1}(x) = x^{1/3}$
#' - Two approaches:
#'     - directly sample from Beta(3,1)
#'     - sample from UNIF(0, 1) and take inverse CDF transformation
#'     
#' Beta distribution (two approaches)
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
set.seed(32611)
n <- 10000
x1 <- rbeta(n, 3, 1)
x2_0 <- runif(n)
x2 <- x2_0^{1/3}

par(mfrow=c(1,2))
hist(x1, nclass=50)
hist(x2, nclass=50)

#' 
#' 
#' 
#' Compare direct sampling and inverse CDF sampling (In class exercise)
#' ===
#' 
#' 1. directly sample n=1000 samples from N(1,2)
#' 2. use inverse CDF method to sample n=1000 from N(1,2) (hint: consider qnorm function)
#' 3. Compare theoritical CDF, empirical CDF from 1 and empirical CDF from 2.
#' 3. Using qq-plot to compare empirical CDF from 1 and empirical CDF from 2.
