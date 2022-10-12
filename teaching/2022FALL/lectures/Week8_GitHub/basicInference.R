#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday Oct 12th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Basic statistical inference in R"
#' ---
#' 
#' 
#' Outlines
#' ===
#' 
#' - Simulating random variables
#' - Basic hypothesis testing
#' - linear regression model
#' 
#' Random samples from a given pool of numbers
#' ===
## -------------------------------------------------------------
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
## -------------------------------------------------------------
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
## -------------------------------------------------------------
set.seed(32611)
sample(1:10,2); sample(1:10,2); sample(1:10,2)

set.seed(32611)
sample(1:10,2); sample(1:10,2); sample(1:10,2)

#' 
#' However, R version may impact the random number
#' ===
#' 
## -------------------------------------------------------------
sessionInfo()
set.seed(32611)
sample(1:10,2)

#' 
#' 
#' Want to sample from a given pool of numbers with replacement
#' ===
## -------------------------------------------------------------
sample(1:10) ## default is without replacement

#' 
## -------------------------------------------------------------
sample(1:10, replace = T) ## with replacement

#' 
## -------------------------------------------------------------
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
#' ![](../figure/normalPlot.png){width=70%}
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
## -------------------------------------------------------------
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
## -------------------------------------------------------------
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
## -------------------------------------------------------------
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
## -------------------------------------------------------------
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
## -------------------------------------------------------------
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
## -------------------------------------------------------------
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
#' 
#' samples from Normal distribution
#' ===
#' 
## -------------------------------------------------------------
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
## -------------------------------------------------------------
set.seed(32608)
z <- rnorm(1000)
plot(ecdf(z), ylab="Distribution", main="Empirical distribution",
     lwd=2, col="red")
curve(pnorm, from = -3, to = 3, lwd=2, add = T)
legend("topleft", legend=c("Empirical distribution", "Actual distribution"),
       lwd=2, col=c("red","black"))

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
## -------------------------------------------------------------
pnorm(q = 2, mean = 0, sd = 1, lower.tail = FALSE) ## default: lower.tail = TRUE

#' 
#'     - two sided test, p-value is 
#'     
## -------------------------------------------------------------
pnorm(q = 2, mean = 0, sd = 1, lower.tail = FALSE) + pnorm(q = - 2, mean = 0, sd = 1, lower.tail = TRUE)

#' 
#' quantile: qnorm()
#' ===
#' 
#' - qnorm() calculates: given the area under the density curve (cumulative density), what is the quantile to generate such cumulative density?
#' 
## -------------------------------------------------------------
qnorm(p = 0.975, mean = 0, sd = 1, lower.tail = TRUE) 
qnorm(p = 0.025, mean = 0, sd = 1, lower.tail = TRUE) 

#' 
#' - this is why `r signif(qnorm(p = 0.975, mean = 0, sd = 1, lower.tail = TRUE), 3)` corresponds to 95% confidence interval.
#' 
#' ![](../figure/ci95.gif)
#' 
#' 
#' Commonly used statistical tests
#' ===
#' 
#' - t.test
#' - wilcox rank test
#' - chi square test
#' - Fisher's exact test
#' - correlation test
#' - KS test
#' 
#' 
#' Two sample t.test
#' ===
#' 
#' - To compare the mean values of two samples that are normally distributed.
#' - Null hypothesis: mean values of two samples are the same.
#' 
## -------------------------------------------------------------
set.seed(32611)
a <- rnorm(10,0,1)
b <- rnorm(10,1,1)
t.test(a,b)

#' 
#' Two sample paired t.test
#' ===
#' 
#' - Used when your data are in matched pairs
#' - Null hypothesis: mean values of two samples are the same.
#' 
## -------------------------------------------------------------
set.seed(32611)
a <- rnorm(10,0,1)
b <- rnorm(10,1,1)
t.test(a,b,paired=TRUE)

#' 
#' 
#' One sample t.test
#' ===
#' 
#' - To test if the group mean is different from 0
#' - Null hypothesis: mean value of a group is 0.
#' 
## -------------------------------------------------------------
set.seed(32611)
a <- rnorm(10,0,1)
t.test(a)

#' 
#' 
#' Two sample Wilcox Rank Sum test
#' ===
#' 
#' - To compare the median values of two samples without any distribution assumption.
#' - non-parametric test
#' - Null hypothesis: median values of two samples are the same.
#' 
## -------------------------------------------------------------
set.seed(32611)
a <- rnorm(10,0,1)
b <- rnorm(10,1,1)
wilcox.test(a,b)

#' 
#' Two sample paired Wilcox Rank Sum test
#' ===
#' 
#' - Used when your data are in matched pairs
#' - Null hypothesis: median values of two samples are the same.
#' 
## -------------------------------------------------------------
set.seed(32611)
a <- rnorm(10,0,1)
b <- rnorm(10,1,1)
wilcox.test(a,b, paired = TRUE)

#' 
#' 
#' 
#' Chi square test
#' ===
#' 
#' ![](../figure/oddsRatioTable.png){width=50%}
#' 
#' - test for independence
#'     - Null hypothesis: the treatment effects are the same between treatment and placebo groups.
#' 
#' 
#' - Consider the following data that describes the relationship between myocardial infarction and aspirin use (Agresti 1996). 
#' 
## -------------------------------------------------------------
X <- matrix(c(189, 104, 10845, 10933), nrow=2,
            dimnames=list(Treatment=c("Placebo","Aspirin"), 
                          "Myocardial Infarction"=c("Yes", "No")))

#' 
#' Chi square test
#' ===
#' 
#' - all of the expected numbers should be greater than 5
#' - Can also be applied to tables with more than 2 categories. 
#' - Null hypothesis: the treatment effects are the same between treatment and placebo groups.
#' 
## -------------------------------------------------------------
X

chisq.test(X)

#' 
#' 
#' Fisher's exact test
#' ===
#' 
#' - Also test for independence
#' - No approximation.
#' - Null hypothesis: the treatment effects are the same between treatment and placebo groups.
#' 
## -------------------------------------------------------------
X

fisher.test(X)

#' 
#' 
#' correlation test
#' ===
#' 
#' - Null hypothesis: there is no correlation between vector a and b.
#' 
## -------------------------------------------------------------
set.seed(32611)
a <- rnorm(10,0,1)
b <- rnorm(10,1,1)

cor.test(a,b) ## parametric test: Gaussian assumption

cor.test(a,b, method = "spearman") ## non-parametric test: no distribution assumption

#' 
#' 
#' KS test
#' ===
#' 
#' - Kolmogorov-Smirnov test: Do x and y come from the same distribution?
#' - Null hypothesis: x and y come from the same distribution.
#' 
## -------------------------------------------------------------
set.seed(32611)
x <- rnorm(50,0,1)
y <- rnorm(50,1,1)
ks.test(x,y)

#' 
#' ```
#' z <- runif(50)
#' ks.test(x,z)
#' ```
#' 
#' 
#' 
#' Fit linear model in R, lm()
#' ===
#' 
#' Basic syntax for lm()
## ---- eval=F--------------------------------------------------
## lm(formula, data)

#' 
#' - formula: Symbolic description of the model
#' - data: Optional dataframe containing the variables in the model
#' - summary() function summarize the linear model result
#' 
## -------------------------------------------------------------
lmfit <- lm(mpg ~ cyl, data=mtcars)
lmfit

#' 
#' ---
#' 
## -------------------------------------------------------------
summary(lmfit)

#' 
#' - Interpretation:
#'   - Unit increase in cyl will result in 2.88 decrease in mpg.
#'   - p-value of cyl < 0.05 indicates cyl is a significant predictor.
#' 
#' 
#' 
#' ---
#' 
#' - outcome variable: mpg
#' - predictors: cyl, disp
#' 
## -------------------------------------------------------------
lmfit <- lm(mpg ~ cyl + disp, data=mtcars)
summary(lmfit)

#' 
#' ---
#' 
#' - outcome variable: mpg
#' - predictors: all the rest of the variables
#' 
## -------------------------------------------------------------
lmfit <- lm(mpg ~ ., data=mtcars)
summary(lmfit)

