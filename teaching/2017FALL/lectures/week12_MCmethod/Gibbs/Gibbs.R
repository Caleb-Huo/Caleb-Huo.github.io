#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Gibbs sampling"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday November 15, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' 
#' Markov Chain Monte Carlo (MCMC)
#' ===
#' 
#' - Metropolis-Hastings algorithm
#'     - Burn-in period
#'     - Monitor convergence
#' 
#' - Question: Since we know samples are correlated from Metropolis-Hastings algorithm, can we still estimate an estimator $\theta$ by sample average?
#'     - Yes, if the underlying density function is also about $\theta$.
#'     - Weak law of large number (WLLN) is still valid if samples are correlated, the only requirement is the samples are identical distributed.
#' 
#' - Another popular algorithm: Gibbs sampling.
#' 
#' 
#' Gibbs Sampling algorithm (Wikipedia)
#' ===
#' - Gibbs sampling is named after the physicist Josiah Willard Gibbs, in reference to an analogy between the sampling algorithm and statistical physics. 
#' - The algorithm was described by brothers Stuart and Donald Geman in 1984, some eight decades after the death of Gibbs.
#' - Josiah Willard Gibbs (February 11, 1839 - April 28, 1903) was an American scientist who made important theoretical contributions to physics, chemistry, and mathematics.
#' 
#' 
#' Gibbs Sampling motivating example
#' ===
#' 
#' ![](../figure/graph/Slide1.png)
#' 
#' - Prior
#'     - $P(I) = 0.5$, $P(-I) = 0.5$
#' - Likelihood (Generative process) for $G$:
#'     - $P(G|I) = 0.8$, $P(-G|I) = 0.2$
#'     - $P(G|-I) = 0.5$, $P(-G|-I) = 0.5$
#' - Likelihood (Generative process) for $S$:
#'     - $P(S|I) = 0.7$, $P(-S|I) = 0.3$
#'     - $P(S|-I) = 0.5$, $P(-S|-I) = 0.5$
#' 
#' Posterior distribution (via Bayes rule)
#' ===
#' 
#' \begin{align*} 
#' P(I | G, S) &= \frac{P(I ,G, S)}{P(G, S)} \\
#' &= \frac{P(G, S|I)P(I)}{P(G, S|I)P(I) + P(G, S|-I)P(-I)} \\
#' &= \frac{P(G|I)P(S|I)P(I)}{P(G|I)P(S|I)P(I) + P(G|-I)P(S|-I)P(-I)} \\
#' &=  \frac{0.8\times 0.7 \times 0.5}{0.8\times 0.7 \times 0.5 + 0.5\times 0.5 \times 0.5}  \\ 
#' &= 0.69
#' \end{align*}
#' 
#' 
#' Similarly
#' 
#' - $P(I | G, -S) = 0.49$
#' - $P(I | -G, S) = 0.36$
#' - $P(I | -G, -S) = 0.19$
#' 
#' Gibbs Sampling algorithm
#' ===
#' 
#' - Suppose the graphical model contains variable $\theta_1, \ldots, \theta_p$.
#' - Initialize starting values for $\theta_1, \ldots, \theta_p$.
#' - Do until convergence:
#'     - Pick an ordering of the $p$ variables (can be fixed or random).
#'     - For each variable $\theta_i$ in order:
#'         - Sample $\theta$ from $P(\theta_i| \theta_1, \ldots,\theta_{i-1}, \theta_{i+1},\ldots \theta_p,X)$, the conditional distribution of $\theta_i$ given the current values of all other variables.
#'         - Update $\theta_i \leftarrow \theta$
#' 
#' 
#' 
#' Gibbs Sampling motivating example
#' ===
#' 
#' 1. Initialize I,G,S.
#' 2. Pick an updating order. (e.g. I,G,S)
#' 3. Update each individual variable given all other variables.
#' 
#' | Iteration  | I    | G     | S  |
#' |-----------:|:-----|:-----:|:--:|
#' |init	       | $1$    | $1$     | $1$  |
#' |1    	     |      |       |    |
#' |2    	     |      |       |    |
#' |3           |      |       |    |
#' |...         | ...     |   ...    |  ...  |
#' |K           |      |       |    |
#' 
#' 
#' Implementation in R
#' ===
#' 
## ------------------------------------------------------------------------
I <- 1; G <- 1; S <- 1
pG <- c(0.5, 0.8)
pS <- c(0.5, 0.5)
pI <- c(0.19, 0.36, 0.49,0.69)

i <- 1
plot(1:3,ylim=c(0,10),type="n", xaxt="n", ylab="iteration")
axis(1, at=1:3,labels=c("I", "G", "S"), col.axis="red")
text(x = 1:3, y= i, label = c(I, G, S))

set.seed(32611)
while(i<10){
  I <- rbinom(1,1,pI[2*G+S+1])
  G <- rbinom(1,1,pG[I+1])
  S <- rbinom(1,1,pS[I+1])
  i <- i + 1  
  text(x = 1:3, y= i, label = c(I, G, S))
}


#' 
#' 
#' Frequentist and Bayesian philosophy
#' ===
#' 
#' - Frequentist:
#'     - parameters $\theta$ are fixed
#'     - data $X$ are random variables
#'     - Goal: create procedures with frequency guarantees
#' 
#' - Bayesian:
#'     - parameters $\theta$ are random variables
#'     - data $X$ are random variables
#'     - analyze beliefs
#' 
#' 
#' An example of linear regression
#' ===
#' 
#' - Underlying truth: $n=100$, $\alpha=0$, $\beta=2$, $\sigma^2=0.5$.
#' - Simulate iid samples: $x_i \sim N(0,1)$, $y_i \sim N(\alpha + \beta x_i,\sigma^2)$. ($i = 1, \ldots, 100$)
#' - Purpose: want to infer $\alpha$, $\beta$, $\sigma^2$.
#' - Set up priors: $\alpha \sim N(\alpha_0, \tau_a^2)$, $\beta \sim N(\beta_0, \tau_b^2)$, $\sigma^2 \sim IG(\nu, \mu)$.
#' $\alpha_0 = 0$, $\beta_0 = 0$, $\tau_a^2 = 10$, $\tau_b^2 = 10$, $\nu=3$, $\mu=3$.
#' - Posterior can be derived using Bayes rule: 
#' 
#'     - $var_\alpha \doteq 1/(1/\tau_a^2 + n/\sigma^2)$, $var_\beta \doteq 1/(1/\tau_b^2 + \sum_{i=1}^n x_i^2/\sigma^2)$
#' 
#'     - $\alpha | x_1^n, y_1^n, \beta, \sigma^2 \sim N(var_\alpha(\sum_{i=1}^n (y_i - \beta x_i)/\sigma^2 + \alpha_0 / \tau_a^2), var_\alpha)$
#' 
#'     - $\beta | x_1^n, y_1^n, \alpha, \sigma^2 \sim N(var_\beta(\sum_{i=1}^n \big( (y_i - \alpha) x_i \big) /\sigma^2 + \beta_0 / \tau_b^2), var_\beta)$
#' 
#'     - $\sigma^2 | x_1^n, y_1^n, \alpha, \beta \sim IG(\nu + n/2, \mu + \sum_{i=1}^n (y_i - \alpha - \beta x_i)^2/2)$
#' 
#' 
#' 
#' Demonstrate the R code
#' ===
#' 
#' - MCMC simulation 1,000 times.
#' - Visualization of MCMC chain
#' - Remove burning 100.
#' - Distribution of parameter.
#' - Auto correlation. (ACF)
#' - Effective samples.
#' - Posterior mean.
#' - chain mixing.
#' 
#' 
#' Demonstrate the R code
#' ===
#' 
## ------------------------------------------------------------------------
set.seed(32611)
n = 100; alpha = 0; beta=2; sig2=0.5;true=c(alpha,beta,sig2)
x=rnorm(n)
y=rnorm(n,alpha+beta*x,sqrt(sig2))
# Prior hyperparameters
alpha0=0;tau2a=10;beta0=0;tau2b=10;nu0=3;s02=1;nu0s02 = nu0*s02
# Setting up starting values
alpha=0;beta=0;sig2=1
# Gibbs sampler
M = 1000
draws = matrix(0,M,3)
draws[1,] <- c(alpha,beta,sig2)
for(i in 2:M){
  var_alpha = 1/(1/tau2a + n/sig2)
  mean = var_alpha*(sum(y-beta*x)/sig2 + alpha0/tau2a)
  alpha = rnorm(1,mean,sqrt(var_alpha))
  var_beta = 1/(1/tau2b + sum(x^2)/sig2)
  mean = var_beta*(sum((y-alpha)*x)/sig2+beta0/tau2b)
  beta = rnorm(1,mean,sqrt(var_beta))
  sig2 = 1/rgamma(1,(nu0+n/2),(nu0s02+sum((y-alpha-beta*x)^2)/2))
  draws[i,] = c(alpha,beta,sig2)
}

#' 
#' 
#' Check Gibbs sampling result
#' ===
#' 
## ------------------------------------------------------------------------
# Markov chain + marginal posterior
names = c('alpha','beta','sig2')
colnames(draws) <- names
ind = 101:M
par(mfrow=c(3,3))
for(i in 1:3){
  ts.plot(draws[,i],xlab='iterations',ylab="",main=names[i])
  abline(v=ind[1],col=4)
  abline(h=true[i],col=2,lwd=2)
  acf(draws[ind,i],main="")
  hist(draws[ind,i],prob=T,main="",xlab="")
  abline(v=true[i],col=2,lwd=2)
}

#' 
#' Posterior mean
#' ===
## ------------------------------------------------------------------------
colMeans(draws[101:M,])

#' 
#' MCMC simulation, number of iterations (e.g. 1,000)
#' ===
#' - Large number of iterations will tend to recover the underlying distribution in higher resolution.
#' - Large number of iterations will also add computational burden.
#' 
#' Burnin period 1
#' ===
#' - If the initial value is within the range of **stationary distribution**, we won't need burnin.
#' - If the initial value is out the range of stationary distribution, we need need to discard them.
#' 
#' ![](../figure/burnin2.png)
#' 
#' 
#' A burnin example from my own research
#' ===
#' ![](../figure/burnin.png)
#' 
#' 
#' Auto correlation
#' ===
#' - MCMC chains always show autocorrelation (AC).
#' - AC means that adjacent samples in time are highly correlated.
#' 
#' $$R_x(k) = \frac
#' {\sum_{t=1}^{n-k} (x_t - \bar{x})(x_{t+k} - \bar{x})}
#' {\sum_{t=1}^{n-k} (x_t - \bar{x})^2}
#' $$
#' 
#' ![](../figure/ACF.png)
#' 
#' - The first-order AC $R_x(1)$ can be used to estimate the Sample Size Inflation Factor (SSIF):
#' $$
#' s_x = \frac{1 + R_x(1)}{1 - R_x(1)}
#' $$
#' - If we took n samples with SSIF, then the effective sample size is $n/s_x$
#' 
#' How to deal with autocorrelation?
#' ===
#' 
#' High autocorrelation leads to smaller effective sample size.
#' 
#' - Design smarter algorithm to make auto-correlation smaller.
#' - Thining: Only take samples every 10 iterations.
#' - Just keep everything and make the Markov chain longer.
#' 
#' Chain mixing
#' ===
#' 
#' - Monitor convergence by plotting samples (of r.v.s) from multiple chains.
#'     - If the chains are well mixed (left), they are probably converged.
#'     - If the chains are poorly-mixed (right), we should continue burn-in.
#' 
#' ![](../figure/chainMixing.png)
#' 
#' Converge
#' ===
#' 
#' - How to monitor the convergence of the Markov chain.
#'     - Monitor the pattern of samples.
#'     - Monitor likelihood.
#'     
#' ![](../figure/converge.png)
#' 
#' 
#' Why Gibbs is MH with $A(x'|x) = 1$?
#' ===
#' 
#' - The Gibbs sampling proposal distribution is:
#' $$q(x'_i, \textbf{x}_{-i} | x_i, \textbf{x}_{-i} ) = p(x'_i | \textbf{x}_{-i})$$
#' - Applying MH to this proposed distribution:
#' 
#' \begin{align*} 
#' A(x'_i, \textbf{x}_{-i} | x_i, \textbf{x}_{-i} ) &= \min 
#' \bigg( 1, 
#' \frac{p(x'_i,\textbf{x}_{-i}) q(x_i, \textbf{x}_{-i} | x'_i, \textbf{x}_{-i} ) }
#' {p(x_i,\textbf{x}_{-i}) q(x'_i, \textbf{x}_{-i} | x_i, \textbf{x}_{-i} )}  
#' \bigg) \\
#' &= \min 
#' \bigg( 1, 
#' \frac{p(x'_i,\textbf{x}_{-i}) p(x_i| \textbf{x}_{-i} ) }
#' {p(x_i,\textbf{x}_{-i}) p(x'_i,  |\textbf{x}_{-i} )}  
#' \bigg) \\
#' &= \min 
#' \bigg( 1, 
#' \frac{p(x'_i| \textbf{x}_{-i} ) p(\textbf{x}_{-i}) p(x_i| \textbf{x}_{-i} ) }
#' {p(x_i| \textbf{x}_{-i} ) p( \textbf{x}_{-i}) p(x'_i  |\textbf{x}_{-i} )}  
#' \bigg) \\
#' &= \min (1,1)\\
#' &= 1
#' \end{align*}
#' 
#' - Gibbs sampling is a MH with acceptance rate 100\%.
#' 
#' 
#' References
#' ===
#' 
#' - BDA3
#' 
## ------------------------------------------------------------------------
knitr::purl("Gibbs.rmd", output = "Gibbs.R ", documentation = 2)

