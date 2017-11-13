#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Monte Carlo methods"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Nov 13, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' 
#' Bayesian sampling method
#' ===
#' 1. Monte Carlo methods. 
#'     - Direct sampling.
#'     - Rejection sampling.
#'     - Importance sampling.
#'     - Sampling-importance resampling.
#' 2. Markov chain Monte Carlo.
#'     - Metropolis-Hasting.
#'     - Gibbs sampling. (Next lecture)
#' 
#' Notation
#' ===
#' 
#' - Denote $p$, $q$ as unnormalized density function.
#'     - E.g. $p(x) = x(1 - x)$, $x \in (0, 1)$.
#' - Denote $p^*$, $q^*$ as normalized density function.
#'     - E.g. $p^*(x) = \frac{p(x)}{\int_x p(x) dx} = 6x(1 - x)$, $x \in (0, 1)$.
#'     - E.g. $q^*(x) = dnorm(x,0,1) = \frac{1}{\sqrt{2\pi}} \exp(-x^2)$.
#' 
#' 
#' Our targets
#' ===
#' 
#' - <span style="color:red">Target 1</span>, To generate Monte Carlo samples $x_m$ from a given probability distribution $p^*(x)$ or $p(x)$.
#' - <span style="color:red">Target 2</span>, To estimate expectations of functions under this distribution, for example
#' $$\mathbb{E}(g (x) | p^*(x)) = \int g(x) p^*(x) dx,$$
#' 
#' Examples: $\mathbb{E} (x | p^*(x))$ or $\mathbb{V}\mbox{ar} (x | p^*(x))$
#' 
#' A simulation approach
#' ===
#' 
#' Problem: We want to estimate $$\mathbb{E}(g (x) | p^*(x)) = \int g(x) p^*(x) dx,$$
#' Given distribution $p^*(x)$.
#' 
#' Examples: $\mathbb{E} (x | p^*(x))$ or $\mathbb{V}\mbox{ar} (x | p^*(x))$
#' 
#' - To generate samples $x_m$ from a given probability distribution $p^*(x)$.
#'     - sample using R: (e.g. rnorm).
#'     - CDF transformation: sample from UNIF(0,1). Then use inverse CDF transformation.
#'     
#' - To estimate expectations of functions under this emperical distribution
#' $$\mathbb{E}(g (x) | p^*(x))  \approx \frac{1}{M} \sum_{m=1}^M g(x^{(m)})$$
#'     - $M$ is number of Monte Carlo samples.
#'     - Via central limit theorem, this is valid.
#' 
#' 
#' Un-normalized density function problems
#' ===
#' - Unnormalized density function: $p(x) = \exp [ 0.4(x-0.4)^2 - 0.08x^4  ]$
#'     - $x \in (-4, 4)$ 
#' 
## ------------------------------------------------------------------------
p <- function(x, a=.4, b=.08){exp(a*(x-a)^2 - b*x^4)}
x <- seq(-4, 4, 0.01)
plot(x,p(x),type="l",  main = expression(p(x) == exp (0.4(x-0.4)^{2}  - 0.08 * x^{4})))

#' 
#' - Hard to compute the normalization factor $Z$.
#' 
## ------------------------------------------------------------------------
integrate(f = p, lower = -4, upper = 4)

#' - Even if we know $Z$, it is still challenge to draw samples.
#' - Direct solution: partition the distribution into bins and direct sampling.
#' 
#' Direct Sampling
#' ===
#' 
#' - Partition the distribution into bins and direct sampling.
#' 
## ------------------------------------------------------------------------
p <- function(x, a=.4, b=.08){exp(a*(x-a)^2 - b*x^4)}
x2 <- seq(-4, 4, 0.1)
plot(x,p(x),type="n",  main = expression(p(x) == exp (0.4(x-0.4)^{2}  - 0.08 * x^{4})))
segments(x2,0,x2,p(x2))

#' 
#' - Diffculty
#'     - Higher-dimensional: say $N = 5$
#'     - For each dimension, we divide the domain equally into $50$ bins
#'     - There will be $50^{5}$ sampling space, huge!
#' 
#' 
#' 
#' Rejection sampling (Demonstrate Monte Carlo method)
#' ===
#' 
#' - $p^*(x)$ is difficult to directly draw samples, but $p(x)$ is easy to evaluate the function value. 
#' 
#' (e.g. $p(x) = \exp [ 0.4(x-0.4)^2 - 0.08x^4  ]$)
#' 
#' - Sample from a simpler distribution $q^*(x)$.
#' Rejection sampling algorithm:
#' 
#' 1. $x \sim q^*(x)$
#' 2. accept $x$ with prob $p(x)/c q^*(x)$:
#'     - Sample $u \sim \mbox{UNIF}(0,1)$, accept if $p(x)/c q^*(x) > u$.
#' 3. Repeat Step 1 and 2 many times.
#' 
#' 
#' Rejection sampling
#' ===
#' 
## ------------------------------------------------------------------------
p <- function(x, a=.4, b=.08){exp(a*(x-a)^2 - b*x^4)}
x <- seq(-4, 4, 0.01)
qstar <- function(x, C = 30){
  C*dnorm(x,sd = 3) 
}
plot(x,p(x),type="l", ylim = c(0,5))
curve(qstar,add = T)
text(0, 4.5, expression({q^"*"} (x) == 30* N (x , 0, 3^2) ))
text(1, 2, expression(p(x) == exp (0.4(x-0.4)^{2}  - 0.08 * x^{4})))
x0 <- -2.5
segments(x0,0,x0,qstar(x0),col=2)
N <- 10
for(i in 1:N){
  set.seed(i)
  ay <- runif(1,0,qstar(x0))
  acol = ifelse(ay < p(x0),2,4)
  points(x0,ay,col=acol,pch=19)
}

#' 
#' 
#' Rejection sampling
#' ===
#' 
#' Proof:
#' \begin{align*} 
#' p^*(x) &= \frac{p(x)}{Z} \\
#' &= \frac{p(x)}{\int_x p(x) dx} \\
#' &=  \frac{[p(x)/c q^*(x)]q^*(x)}{\int_x [p(x)/c q^*(x)]q^*(x)dx} \\ 
#' \end{align*}
#' 
#' Interpretation of the numerator:
#' 
#' - $q^*(x):$ Sampling from the proposed distribution.
#' - $p(x)/c q^*(x):$ Rejection probability.
#' 
#' Rejection sampling
#' ===
## ------------------------------------------------------------------------
## rejection sampling
p <- function(x, a=.4, b=.08){exp(a*(x-a)^2 - b*x^4)}
x <- seq(-4, 4, 0.1)
plot(x,p(x),type="l")
# uniform proposal on [-4,4]:
qstar <- function(x){rep.int(0.125,length(x))}
# we can find M in this case:
C <- round(max(p(x)/qstar(x))) + 1; C
# number of samples
N <- 1000
# generate proposals and u
x.h <- runif( N, -4, 4 )
u <- runif( N )
acc <- u < p(x.h) / (C * qstar(x.h))
x.acc <- x.h[ acc ]
# how many proposals are accepted
sum( acc ) /N

#' 
#' ---
#' 
## ------------------------------------------------------------------------
# calculate some statistics
c(m=mean(x.acc), s=sd(x.acc))

par(mfrow=c(1,2), mar=c(2,2,1,1))
plot(x,p(x),type="l")
barplot(table(round(x.acc,1))/length(x.acc))

#' 
#' **Discussion:** What does the acceptance rate depend on?
#' 
#' 
#' Importance sampling
#' ===
#' 
#' Importance sampling is not a method for generating samples from $p(x)$ (target 1),
#' it is just a method for estimating the expectation of a function $g(x)$ (target 2).
#' 
#' - Sampling from $p^*(x)$ is hard.
#' - Want to calculate expectation of $\phi(x)$ under $p^*(x)$.
#' - Suppose we can sample from a simpler proposal distribution $q^*$ instead.
#' - If $q^*$ dominates $p^*$ (i.e., $q^*(x)>0$ whenever $p^*(x)>0$), we can sample from $q^*$ and reweight: $w(x) = \frac{p^*(x)}{q^*(x)}$
#' 
#' ![](../figure/importanceSamplingAlgorithm.png)
#' 
#' 
#' Importance sampling, algorithm
#' ===
#' - Underlying distribution: $p(x)$ or $p^*(x) = \frac{p(x)}{Z}$.
#' - Proposed distribution (Sampler): $q^*(x)$.
#' - Function of interest $\phi (x)$.
#' 
#' \begin{align*} 
#' \mathbb{E} (\phi (x) | p^* )  &= \int \phi (x) p^*(x) dx\\ 
#' &= \frac{\int \phi (x) p^*(x) dx}{\int p^*(x) dx}\\ 
#' &=  \frac{\int [\phi (x) p(x)/Z] dx}{\int [p(x)/Z] dx}\\ 
#' &=  \frac{\int [\phi (x) p(x)/q^*(x)] q^*(x) dx}{\int [p(x)/q^*(x)]  q^*(x) dx},
#' \end{align*}
#' 
#' - This can be estimated using M draws $x_1, \ldots, x_M$ from $q^*(x)$ by the following expression.
#' 
#' $$\hat{\mathbb{E}} (\phi (x) | p^* ) = \frac{\frac{1}{M} \sum_{m=1}^M[\phi (x_m) p(x_m)/q^*(x_m)] }{ \frac{1}{M} \sum_{m=1}^M[p(x_m)/q^*(x_m)]}$$
#' 
#' 
#' $w(x_m) =\frac{p(x_m)}{q^*(x_m)}$
#' 
#' $$\hat{\mathbb{E}} (\phi (x) | p^* )  = \frac{ \sum_{m=1}^M \phi(x_m) w(x_m)}{ \sum_{m=1}^M w(x_m)}
#' $$
#' 
#' - Importance ratio: $\frac{w(x_m)}{ \sum_{m=1}^M w(x_m)}$
#' 
#' 
#' Importance sampling, examples
#' ===
#' 
#' - Problem setting:
#' 
## ------------------------------------------------------------------------
par(mfrow=c(1,2), mar=c(2,2,2,1))

p <- function(x, a=.4, b=.08){exp(a*(x-a)^2 - b*x^4)}
x <- seq(-4, 4, 0.01)
plot(x,p(x),type="l",  main = expression(p(x) == exp (0.4(x-0.4)^{2}  - 0.08 * x^{4})))

phi <- function(x){ (- 1/3*x^3 + 1/2*x^2 + 12*x - 12) / 30 + 1.3}
x <- seq(-4, 4, 0.01)
plot(x,phi(x),type="l",main= expression(phi(x)))

#' 
#' - $p(x) = \exp [0.4 (x - 0.4) ^ 2 - 0.08 x^4]$
#' - $\phi (x) = (- 1/3x^3 + 1/2x^2 + 12x - 12) / 30 + 1.3$ = right panel.
#' 
#' Underlying solution
#' ===
#' 
#' \begin{align*} 
#' \mathbb{E} (\phi (x) | p^* )  &= \int \phi (x) p^*(x) dx\\ 
#' &= \frac{\int \phi (x) p^*(x) dx}{\int p^*(x) dx}\\ 
#' &=  \frac{\int [\phi (x) p(x)/Z] dx}{\int [p(x)/Z] dx}\\ 
#' &=  \frac{\int [\phi (x) p(x)] dx}{\int [p(x)] dx}\\ 
#' \end{align*}
#' 
#' 
## ------------------------------------------------------------------------
ep <- function(x) p(x)*phi(x)
truthE <- integrate(f = ep, lower = -4, upper = 4)$value/integrate(f = p, lower = -4, upper = 4)$value
truthE

#' 
#' Importance sampling, examples (2)
#' ===
#' 
## ------------------------------------------------------------------------
q.r <- rnorm
q.d <- dnorm

par(mfrow=c(1,2))
plot(x,q.d(x),type="l",main='sampler distribution Gaussian')
curve(p, from = -4,to = 4 ,col=2 ,  main = expression(p(x) == exp (0.4(x-0.4)^{2}  - 0.08 * x^{4})))

M <- 1000
x.m <- q.r(M)
ww <- p(x.m) / q.d(x.m)
qq <- ww / sum(ww)
x.g <- phi(x.m)
sum(x.g * qq)/sum(qq)

#' 
#' Number of samples for importance sampling
#' ===
#' 
## ---- cache=T------------------------------------------------------------
M <- 10^seq(1,7,length.out = 30)

result.g <- numeric(length(M))
for(i in 1:length(M)){
  aM <- M[i]
  x.m <- q.r(aM)
  ww <- p(x.m) / q.d(x.m)
  
  qq.g <- ww / sum(ww)
  x.g <- phi(x.m)
  
  result.g[i] <- sum(x.g * qq.g)/sum(qq.g)
}

plot(log10(M),result.g,main='importance sampling result Gaussian')
abline(h = truthE, col = 2)

#' 
#' Sampling from a narrow Gaussian distribution
#' ===
#' 
## ------------------------------------------------------------------------
q.r_narrow <- function(x){rnorm(x,0,1/2)}
q.d_narrow <- function(x){dnorm(x,0,1/2)}

par(mfrow=c(1,2))
plot(x,q.d_narrow(x),type="l",main='sampler narrow distribution Gaussian')
curve(p, from = -4,to = 4 ,col=2 ,  main = expression(p(x) == exp (0.4(x-0.4)^{2}  - 0.08 * x^{4})))

#' 
#' 
#' Number of samples for importance sampling (narrow Gaussian distribution)
#' ===
#' 
#' 
## ---- cache=T------------------------------------------------------------
M <- 10^seq(1,7,length.out = 30)

result.narrow <- numeric(length(M))
for(i in 1:length(M)){
  aM <- M[i]
  x.m <- q.r_narrow(aM)
  ww <- p(x.m) / q.d_narrow(x.m)
  
  qq.c <- ww / sum(ww)
  x.c <- phi(x.m)
  
  result.narrow[i] <- sum(x.c * qq.c)/sum(qq.c)
}
plot(log(M,10),result.narrow)
abline(h = truthE, col = 2)

#' 
#' Importance sampling remark
#' ===
#' 
#' - Want to estimate the $\mathbb{E} (\phi (x) | p^* )$.
#' - If the proposal density $q^*(x)$ is small in a region where $|\phi(x) p^*(x)|$ is large,
#' it is quite possible that after many points $x_m$ have been generated, none of them fell in that region.
#' This leads to a wrong estimate of $\mathbb{E} (\phi (x) | p^* )$.
#' - Importance sampler should have heavy tails.
#' - If $q^* (x)$ can be chosen such that $\frac{\phi p}{q^*}$ is roughly constant,
#' then fairly precise estimates of the integral can be obtained.
#' - Importance sampling is not a useful method if the importance ratios vary substantially. The worst possible scenario occurs when the importance ratios are small with high probability but with a low probability are huge.
#' 
#' 
#' Importance resampling (SIR)
#' ===
#' 
#' - SIR: sampling-importance resampling.
#' - This is an alternative when rejection sampling constant $c$ is not immediately available.
#' - Algorithm (BDA3, reference)
#'     - Draw samples $x_1, \ldots, x_M \sim q^*(x)$.
#'     - Calculated importance weights $w_m = p(x_i)/q^*(x_i)$.
#'     - Normalize the weights  as $W_m = \frac{w_m}{\sum_m w_m}$ (importance ratio).
#'     - Resample ($K$ out of $M$) from $\{ x_1, \ldots, x_M \}$ where $y_k, 1\le k \le K$ is drawn with probability $W_m$. (without replacement)
#' 
#' Remark:
#' 
#' - also see other people Resample ($M$ out of $M$) with replacement.
#' 
#' 
#' Implement importance resampling (SIR)
#' ===
## ------------------------------------------------------------------------
p <- function(x, a=.4, b=.08){exp(a*(x-a)^2 - b*x^4)}
x <- seq(-4, 4, 0.01)
plot(x,p(x),type="l")

qstar <- function(x){rep.int(0.125,length(x))}
N <- 10000
S <- 1000
x.qstar <- runif( N, -4, 4 )
ww <- p(x.qstar) / qstar(x.qstar)
qq <- ww / sum(ww)
x.acc <-sample(x.qstar, size = S, prob=qq, replace=F)
par(mfrow=c(1,2), mar=c(2,2,1,1))
plot(x,p(x),type="l")
barplot(table(round(x.acc,1))/length(x.acc))

#' 
#' 
#' Summarize
#' ===
#' - Direct sampling. (target 1)
#' - Rejection sampling. (target 1)
#' - Importance sampling. (target 2)
#' - Sampling-importance resampling. (target 1)
#' 
#' 
#' 
#' limitation of Monte Carlo method
#' ===
#' - Direct sampling
#'     - Often hard to compute the normalization factor $Z$.
#'     - Hard to get rare events, especially in higher dimensional spaces.
#' - Rejection sampling, importance sampling.
#'     - Do not work well if the proposed distribution $q^*(x)$ is very different from $p(x)$.
#'     - Constructing a $q^*(x)$ similar to $p(x)$ can be difficult.
#'         -  Making a good proposal usually requires knowledge of the analytic form of $p(x)$ - but if we had that, we wouldn't even need to sample!
#' - Solution: instead of a fixed proposed distribution $q^*(x)$, we can use an adaptive proposal.   
#'         
#' 
#' 
#' Motivation of Metropolis-Hastings
#' ===
#' 
#' - Drawbacks of rejection sampling and SIR are that it is difficult to propose an efficient distribution $q^*(x)$
#' - For rejection sampling, it is also difficult to find $M$.
#' - A smart idea is let the proposed distribution depends on the last accepted value.
#' - Instead of fixed $q(x')$, we use $q(x'|x)$ where $x'$ is the new state being sampled and $x$ is the previous sample.
#' - As $x$ changes, $q(x'|x)$ can also change (as a function of $x'$.)
#' 
#' ![compare importance sampling and Metropolis-Hastings](../figure/compareMHandDS.png)
#' 
#' 
#' 
#' Monte Carlo vs Markov Chain Monte Carlo
#' ===
#' 
#' - Monte Carlo methods: simulation/sampling methods.
#'     - Simulation
#'     - Rejection sampling
#'     - SIR
#' - Markov Chain Monte Carlo: A type of Monte Carlo method -- **next step samples depend on previous samples**.
#'     - Metropolis-Hastings
#'     - Gibbs Sampling
#' 
#' 
#' Introduce MH algorithm
#' ===
#' 
#' - ![compare importance sampling and Metropolis-Hastings](../figure/compareMHandDS.png)
#' 
#' - Draw a sample $x'$ from $q(x'|x)$, where $x$ is the previous sample.
#' - The new sample $x'$ is accepted or rejected with some probability $A(x'|x)$.
#' $$A(x'|x) = \min \bigg(1, \frac{p(x')q(x|x')}{p(x)q(x'|x)} \bigg)$$ 
#' - $A(x'|x) = \min \bigg(1, \frac{p(x')}{q(x'|x)}/\frac{p(x)}{q(x|x')}  \bigg)$ is a ratio of importance sampling weights.
#' 
#' 
#' MH algorithm
#' ===
#' 
#' - Initialize our parameters $x^0$.
#' - Given an accepted value $x^{t-1}$
#'     1. Draw $x \sim q(\cdot|x^{t-1})$.
#'     2. Accept $x$ with probability $A(x|x^{t-1}) = \min \bigg(1, \frac{p(x)q(x^{t-1}|x)}{p(x^{t-1})q(x|x^{t-1})} \bigg)$, 
#' and if accepted, set $x^t = x$.
#'         - Draw $u \sim UNIF(0,1)$.
#'         - If $u < A(x|x^{t-1})$, accept.
#'     3. If we didn't accept $x$, set $x^t = x^{t - 1}$
#' - We repeat $T$ times ($t = 1, \ldots, T$).
#' 
#' 
#' Implementation of MH algorithm
#' ===
#' 
#' - target distribution: $p(x) = \exp [ 0.4(x-0.4)^2 - 0.08x^4  ]$
#' - sampling distribution: $q(x^{(t)}) = dnorm(x^{(t)},x^{(t-1)},1)$
#' 
## ------------------------------------------------------------------------
p <- function(x, a=.4, b=.08){exp(a*(x-a)^2 - b*x^4)}
x <- seq(-4, 4, 0.1)
plot(x,p(x),type="l")

#' 
#' ---
#' 
## ------------------------------------------------------------------------
N <- 10000
x.acc5 <- rep.int(NA, N)
u <- runif(N)
acc.count <- 0
std <- 1 ## Spread of proposal distribution
xc <- 0; ## Starting value
for (ii in 1:N){
  xp <- rnorm(1, mean=xc, sd=std) ## proposal
  alpha <- min(1, (p(xp)/p(xc)) *
                 (dnorm(xc, mean=xp,sd=std)/dnorm(xp, mean=xc,sd=std)))
  x.acc5[ii] <- xc <- ifelse(u[ii] < alpha, xp, xc)
  ## find number of acccepted proposals:
  acc.count <- acc.count + (u[ii] < alpha)
}
## Fraction of accepted *new* proposals
acc.count/N

#' ---
#' 
#' 
## ------------------------------------------------------------------------
par(mfrow=c(1,2), mar=c(2,2,1,1))
plot(x,p(x),type="l")
barplot(table(round(x.acc5,1))/length(x.acc5))

#' 
#' 
#' Check samples from MH
#' ===
#' 
## ------------------------------------------------------------------------
plot(x.acc5,type="l")

#' 
#' Good convergence.
#' 
#' Check samples from MH
#' ===
#' 
#' - burnin period: disgard intial samples since they may not be in the stationary distribution
#'     - initial x = 8?
#' 
#' - play with variance? (doesn't converge)
#'     - sd = 0.1
#'     - sd = 0.01
#' 
#' - How many samples (empirically):
#'     - total 10,000 samples.
#'     - 500 burnin samples.
#' 
#' 
#' Remove initial values for burnin period
#' ===
#' 
## ------------------------------------------------------------------------
N <- 1000
x.acc5 <- rep.int(NA, N)
u <- runif(N)
acc.count <- 0
std <- 1 ## Spread of proposal distribution
xc <- 8; ## Starting value
for (ii in 1:N){
  xp <- rnorm(1, mean=xc, sd=std) ## proposal
  alpha <- min(1, (p(xp)/p(xc)) *
                 (dnorm(xc, mean=xp,sd=std)/dnorm(xp, mean=xc,sd=std)))
  x.acc5[ii] <- xc <- ifelse(u[ii] < alpha, xp, xc)
  ## find number of acccepted proposals:
  acc.count <- acc.count + (u[ii] < alpha)
}
## Fraction of accepted *new* proposals
acc.count/N

## ------------------------------------------------------------------------
plot(x.acc5,type="l")

#' 
#' 
#' Doesn't converge 
#' ===
#' 
## ------------------------------------------------------------------------
N <- 1000
x.acc5 <- rep.int(NA, N)
u <- runif(N)
acc.count <- 0
std <- 0.1 ## Spread of proposal distribution
xc <- 0; ## Starting value
for (ii in 1:N){
  xp <- rnorm(1, mean=xc, sd=std) ## proposal
  alpha <- min(1, (p(xp)/p(xc)) *
                 (dnorm(xc, mean=xp,sd=std)/dnorm(xp, mean=xc,sd=std)))
  x.acc5[ii] <- xc <- ifelse(u[ii] < alpha, xp, xc)
  ## find number of acccepted proposals:
  acc.count <- acc.count + (u[ii] < alpha)
}
## Fraction of accepted *new* proposals
acc.count/N

## ------------------------------------------------------------------------
plot(x.acc5,type="l")

#' 
#' 
#' Why MH converge? (optional)
#' ===
#' 
#' - In MH we draw sample $x'$ according to $q(x'|x)$, then we accept/reject according to $A(x'|x)$.
#' - The transition kernel is $T(x'|x) = q(x'|x) A(x'|x)$.
#' - $A(x'|x) = \min \bigg(1, \frac{p(x')q(x|x')}{p(x)q(x'|x)} \bigg)$
#' - If $A(x'|x) \le 1$, then $\frac{p(x')q(x|x')}{p(x)q(x'|x)} \le 1$, $\frac{p(x)q(x'|x)}{p(x')q(x|x')} \ge 1$, $A(x|x') = 1$.
#' - Proof:
#' $$ A(x'|x) = \frac{p(x')q(x|x')}{p(x)q(x'|x)}$$
#' $$ p(x)q(x'|x) A(x'|x) = p(x')q(x|x') $$
#' $$ p(x)q(x'|x) A(x'|x) = p(x')q(x|x') A(x|x')$$
#' $$ p(x)T(x'|x)  = p(x')T(x|x') $$
#' The last line is called detailed balance condition.
#' 
#' $$ \int_{x} p(x)T(x'|x) dx = \int_{x} p(x')T(x|x') dx$$
#' $$ p(x') = \int_{x} p(x)T(x'|x) dx$$
#' 
#' Since p(x) is the true distribution.
#' MH algorithm will eventually converges to the true distribution.
#' 
#' 
#' Special cases for MH
#' ===
#' 
#' - Metropolis algorithm: 
#'     - The proposed distribution is symmetrical, e.g. $q(x'|x) = q(x|x')$ for all pairs (x,x'). In this case the acceptance probability is $A(x'|x) = \min (1, \frac{p(x')}{p(x)})$
#' 
#' - Random-walk Metropolis: 
#'     - A popular choice for proposal in a Metropolis algorithm is $q(x'|x) = g(x-x')$ where g is symmetric. 
#' 
#' - Independence sampler:
#'     - The proposed distribution $q(x'|x) = q(x')$ doesn't depend on $x$. The acceptance probability becomes $A(x'|x) = \min(1,  \frac{p(x')}{p(x)} \frac{q(x)}{q(x')})$. This works well when $q$ is a good approximation to $p$.
#'     
#' - Gibbs sampling:
#'     - $q(x'|x)$ is the conditional probability given all other variables.  $A(x'|x) = 1$.
#' 
#' Always remember for MCMC
#' ===
#' 
#' - Burn-in period.
#' - Monitor convergence.
#' 
#' 
#' References
#' ===
#' 
#' - [Bayesian Data analysis](https://www.math.muni.cz/~kolacek/docs/bayesian_data_analysis.pdf) Chapter 10-12
#' 
## ------------------------------------------------------------------------
knitr::purl("MC.rmd", output = "MC.R ", documentation = 2)

