#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday October 26, 2020"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Generalized Linear Model
#' ---
#' 
#' 
#' Outline
#' ===
#' 
#' - Generalized Linear Model (GLM)
#'     - Linear regression model is a special case
#'     - Exponential family
#' - Logistic regression
#' - Multinomial logistic regression 
#' - Ordinal logistic regression 
#' - Poisson regression
#' - Negative binomial regression
#' - Zero inflated Poission and negative binomial regression
#' 
#' 
#' Review: linear regression model
#' ===
#' - $Y \in \mathbb{R}$, $X_j \in \mathbb{R}$ with $j = \{1, \ldots, p\}$, $\beta_j \in \mathbb{R}$ with $j = \{0, \ldots, p\}$, $\varepsilon \in \mathbb{R}$
#' 
#' $$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_pX_p + \varepsilon, \text{with }\varepsilon \sim N(0, \sigma^2)$$
#' 
#' - Y: outcome variable or response.
#' - $X_1, X_2, \ldots, X_p$ are called predictors.
#' - $\beta_0, \beta_1, \ldots, \beta_p$ are called coefficient, for which we will estimate.
#' 
#' Intepretation of linear model coefficients:
#' 
#' - $\beta_1$: unit increase of $X_1$ will result in $\beta_1$ increase in $Y$.
#' - p-value of $\beta_1$ is significant: $X_1$ is a significant predictor to explain the varaince in $Y$ adjusting for the other covaraites.
#' 
#' 
#' Limitation of linear model:
#' ===
#' 
#' - $\varepsilon$ has to be normally distributed.
#' - $Y$ has to be a linear function of $X^\top\beta$.
#' 
#' How about other data types?
#' 
#' - Binary data
#'   - 0, 1
#' - Count data
#'   - 0, 4, 12, ...
#' - Oridinal data
#'   - 1,2,3,...
#' 
#' 
#' Generalize the "linear regression model" (1)
#' ===
#' $$Y = X^\top\beta + \varepsilon, \text{with }\varepsilon \sim N(0, \sigma^2)$$
#' 
#' - Random (stochastic) component
#'     - specifies the conditional distribution of the response variable $Y$ given the explanatory variables $X$
#'     - $\mu = E(Y|X), Y|X \sim N(\mu, \sigma^2)$
#' - Systematic component 
#'     – the covariates $X$ combine linearly with the coefficients to form the linear predictor.
#'     - $X^\top\beta$
#' - Link function
#'     - $X^\top\beta = \mu$
#' 
#' Generalize the "linear regression model" (2)
#' ===
#' We can generalize these conditions for linear regression model
#' 
#' - Random (stochastic) component
#'     - specifies the conditional distribution of the response variable $Y$ given the explanatory variables $X$
#'     - $\mu = E(Y|X), Y|X \sim f(\mu)$, $f$ is a distribution from the exponential family.
#'     - In the linear model case, $f$ is Normal distribution.
#' - Systematic component 
#'     – Still assume covariates $X$ combine linearly with the coefficients to form the linear predictor.
#'     - $X^\top\beta$
#' - Link function
#'     - an invertible, monotone and twice-differentiable function $g$ which transforms the expectation of the response to the linear predictor
#'     - $X^\top\beta = g(\mu)$
#'     - In the linear model case, $g$ is an identity function.
#' 
#' 
#' 
#' Commonly used distribution $f$ in GLM
#' ===
#' 
#' - Gaussian distribution: for continuous data
#' - Binomial distribution: for binary data
#' - Possion distribution: for count data
#' 
#' Standard link functions and their inverses
#' ===
#' 
#' Link    | $X^\top\beta = g(\mu)$ | $\mu = g^{-1} (X^\top\beta)$
#' ------------ | -------------- | -------------- 
#' identity | $X^\top\beta = \mu$                |   $\mu = X^\top\beta$
#' log     | $X^\top\beta = \log(\mu)$                |   $\mu = \exp(X^\top\beta)$
#' logit     | $X^\top\beta = \log \frac{\mu}{1 - \mu}$ |   $\mu = \frac{\exp(X^\top\beta)}{1 + \exp(X^\top\beta)}$
#' probit    | $X^\top\beta = \Phi^{-1}({\mu})$                |   $\mu = \Phi (X^\top\beta)$
#' 
#' - $\Phi$: CDF of standard normal distribution.
#' 
#' 
#' logistic regression motivating example
#' ===
#' 
## ------------------------------------------------------------------------
X <- matrix(c(189, 104, 10845, 10933), nrow=2,
            dimnames=list(Treatment=c("Placebo","Aspirin"), 
                          "Myocardial Infarction"=c("Yes", "No")))
X


odds.ratio0 <- function(X){
  result <- X[1,1]*X[2,2]/(X[1,2]*X[2,1])
  return(result)
}
odds.ratio0(X)

#' 
#' logistic regression motivating example
#' ===
#' 
#' - The input data is usually formatted as this way:
#' 
#' ID        | Myocardial Infarction (Yes/No)   | Treatment(1/0)
#' ----------|----------|----------
#' 1 | 1 | 0
#' 2 | 0 | 1
#' 3 | 1 | 1
#' 4 | 1 | 0
#' ... | ... | ...
#' 
#' logistic regression
#' ===
#' 
#' - Random component: $Y|X \sim Bern(\mu)$, Y is binary, either 0 or 1.
#'     - PDF of Bernoulli distribution: $f_\mu(y) = \mu^y(1 - \mu)^{1-y}$
#'       - $0 < \mu < 1$
#'     - $p(Y = 1 | X) = \mu$
#'     - $p(Y = 0 | X) = 1 - \mu$
#'     - $E(Y|X) = 0 \times (1 - \mu) + 1 \times \mu = \mu$
#' - Link function:
#'     - link $\mu$ and $X^\top \beta$: logit link.
#'     - logit link: $logit(\mu) = \log \frac{\mu}{1 - \mu}$
#'     - $X^\top \beta = \log \frac{\mu}{1 - \mu}$
#'     - $\mu = E(Y|X) = \frac{\exp (X^\top \beta)}{1 + \exp (X^\top \beta)}$
#' 
#' 
#' Prostate cancer data
#' ===
#' The data is from the book element of statistical learning
#' 
## ------------------------------------------------------------------------
library(ElemStatLearn)
str(prostate)
prostate$train <- NULL

#' 
#' fit logistic regression in R (logit link)
#' ===
#' 
## ------------------------------------------------------------------------
glm_binomial_logit <- glm(svi ~ lcavol, data = prostate,  family = binomial(link = "logit"))
summary(glm_binomial_logit)

#' 
#' 
#' 
#' fit logistic regression in R (probit link)
#' ===
#' 
## ------------------------------------------------------------------------
glm_binomial_probit <- glm(svi ~ lcavol, data = prostate,  family = binomial(link = "probit"))
summary(glm_binomial_probit)

#' 
#' 
#' Compare logit link and probit link
#' ===
#' 
## ------------------------------------------------------------------------
flogit <- function(x) exp(x)/(exp(x) + 1)
fprobit <- pnorm 

curve(flogit, -5, 5, ylab = "f(x)", lwd = 2)
curve(fprobit, -5, 5, add = T, col = 2 , lwd = 2)
legend("topleft", c("logit", "probit"), col=c(1,2),lwd = 2)

#' 
#' - logit link function is more spread than probit link function.
#' - both of them are useful
#' - Which one is natural?
#' 
#' Exponential family (1)
#' ===
#' 
#' - Generalized liner model (GLM) allows the user to select a distribution from the **exponential family**.
#' - The exponential family comprises a set of flexible distribution ranging both continuous and discrete random varaibles.
#' - Many of probability distributions that we commonly used are specific cases of this family.
#' 
#' 
#' Exponential family (2)
#' ===
#' 
#' - The natural form of the probability density functuion (pdf) of a distribution in the exponential family is
#' $$f(y) = \exp (\frac{y\theta - b(\theta)}{a(\phi)} + c(y; \phi))$$
#' 
#'     - $\theta$ is called the canonical parameter or location parameter of the distribution.
#'     - $\phi$ is called the dispersion parameter or the scale parameter of the distribution, and is constant for all $y$.
#'     - $a(\cdot), b(\cdot)$ and $c(\cdot)$ are known functions.
#' 
#' The role of $b(\theta)$ in the exponential family
#' ===
#' 
#' - $E(Y|X) = \mu = b'(\theta)$
#' - $Var(Y|X) = a(\phi)b''(\theta)$
#' 
#' Comments:
#' 
#' - The first derivative of $b(\theta)$ is the mean of the distribution.
#' - The second derivative of $b(\theta)$ is related to the variance of the distribution.
#' 
#' Prove $E(Y|X) = b'(\theta)$
#' ===
#' 
#' - $f(y; \theta) = \exp (\frac{y\theta - b(\theta)}{a(\phi)} + c(y; \phi))$
#' - $\int f(y; \theta) dy = 1$
#' - $\frac{\partial}{\partial \theta}\int f(y; \theta) dy = 0$
#' - $\int \frac{\partial}{\partial \theta}f(y; \theta) dy = 0$
#' - $\frac{\partial}{\partial \theta}f(y; \theta) = f(y; \theta) \frac{y - b'(\theta)}{a}$
#' - $\int f(y; \theta) \frac{y - b'(\theta)}{a} dy = 0$
#' - $\int y f(y; \theta) dy = \int b'(\theta) f(y; \theta) dy$
#' - $E(Y) = b'(\theta) \int f(y; \theta) dy = b'(\theta)$
#' 
#' 
#' Prove $Var(Y|X) = a(\phi) b''(\theta)$
#' ===
#' 
#' - $f(y; \theta) = \exp (\frac{y\theta - b(\theta)}{a(\phi)} + c(y; \phi))$
#' - $\int f(y; \theta) dy = 1$
#' - $\frac{\partial}{\partial \theta}\int f(y; \theta) dy = 0$
#' - $\int \frac{\partial}{\partial \theta}f(y; \theta) dy = 0$
#' - $\int \frac{\partial^2}{\partial \theta^2}f(y; \theta) dy = 0$
#' - $\frac{\partial}{\partial \theta}f(y; \theta) = f(y; \theta) \frac{y - b'(\theta)}{a}$
#' - $\frac{\partial^2}{\partial \theta^2}f(y; \theta) = f(y; \theta) (\frac{y - b'(\theta)}{a})^2 - f(y; \theta) \frac{b''(\theta)}{a}$
#' - $\int (f(y; \theta) (\frac{y - b'(\theta)}{a})^2 - f(y; \theta) \frac{b''(\theta)}{a}) dy = 0$
#' - $\int f(y; \theta) (\frac{y - b'(\theta)}{a})^2 dy = \int  f(y; \theta) \frac{b''(\theta)}{a} dy$
#' - $\int f(y; \theta) (y - E(Y))^2 dy = a b''(\theta) \int  f(y; \theta) dy$
#' - $Var(Y) = a b''(\theta)$
#' 
#' 
#' Canonical link
#' ===
#' 
#' - Canonical link is a function that links canonical parameter $\theta$ in terms of the mean of the distribution $\mu = E(Y|X)$
#' 
#' - The canonical link for each family is the one most commonly used
#' 
#' - It arises naturally from the general formula for distributions in the exponential families.
#' 
#' $$f(y) = \exp (\frac{y\theta - b(\theta)}{a(\phi)} + c(y; \phi))$$
#' 
#' Canonical link (normal distribution)
#' ===
#' 
#' \begin{aligned}
#' 
#' f(y) &= \exp (\frac{y\theta - b(\theta)}{a(\phi)} + c(y; \phi)) \\
#'       &= \frac{1}{\sqrt{2\pi \sigma^2}} \exp(- \frac{-(y - \mu)^2}{2\sigma^2}) \\
#'       &= \exp( \frac{y \mu - \frac{\mu^2}{2}}{\sigma^2} - \frac{1}{2} [\frac{y^2}{\sigma^2 + \log(2\pi \sigma^2)}] )
#' \end{aligned}
#' 
#' - Canonical parameter $\theta = \mu$
#' 
#' Canonical link (binomial distribution)
#' ===
#' 
#' 
#' $$f(y) = \exp (\frac{y\theta - b(\theta)}{a(\phi)} + c(y; \phi))$$
#' $$f(y) = \mu^y (1 - \mu)^{1 - y}$$
#'     
#' $$f(y) = \exp(\frac{y \log(\frac{\mu}{1 - \mu}) - \log(\frac{1}{1 - \mu})}{1} + 0)$$
#' 
#' - Canonical parameter $\theta = \log(\frac{\mu}{1 - \mu})$
#' 
#' 
#' Canonical link (Poisson distribution)
#' ===
#' 
#' $$f(y) = \exp (\frac{y\theta - b(\theta)}{a(\phi)} + c(y; \phi))$$
#' $$f(y) = \frac{\mu^y \exp (-\mu)}{y!}$$
#'     
#' $$f(y) = \exp(\frac{y \log(\mu)  - \mu}{1} - \log(y!))$$
#' 
#' - Canonical parameter $\theta = \log(\mu)$
#' 
#' 
#' 
#' Canonical link functions for distributions in the exponential families
#' ===
#' 
#' Family    | Canonical link | Canonical parameter $\theta$ | Link function | Mean function
#' ------------ | -------------- | -------------- | -------------- | -------------- 
#' Gaussian | identity | $\theta = \mu$  |   $X^\top\beta = \mu$ |  $\mu = X^\top\beta$
#' Binomial | logit | $\theta = \log(\frac{\mu}{1 - \mu})$  |   $X^\top\beta = \log(\frac{\mu}{1 - \mu})$ |  $\mu = \frac{\exp (X^\top\beta)}{1 + \exp (X^\top\beta)}$
#' Poisson | log | $\theta = \log(\mu)$  |   $X^\top\beta = \log(\mu)$ |  $\mu = \exp(X^\top\beta)$
#' 
#' 
#' fit logistic regression in R (by default using logit link)
#' ===
#' 
## ------------------------------------------------------------------------
glm_binomial_logit <- glm(svi ~ lcavol, data = prostate,  family = binomial())
summary(glm_binomial_logit)

#' 
#' 
#' interpret logistic regression coefficient $\beta$
#' ===
#' 
#' - $logit(\mu_i) = \log(\frac{\mu_i}{1 - \mu_i}) = X_i^\top \beta = \beta_0 + \beta_1 x_i$
#' - $\frac{\mu_i}{1 - \mu_i} = \exp(\beta_0 + \beta_1 x_i)$
#' - Suppose there is only one binary predictor x
#' 
#' $$\hat{OR} = \frac{\hat{odds}|_{x_i=1}}{\hat{odds}|_{x_i=0}}
#' = \frac{\frac{P(y_i=1)}{P(y_i=0)}|_{x_i=1}}{\frac{P(y_i=1)}{P(y_i=0)}|_{x_i=0}}
#' = \frac{\frac{\hat{\mu}_i}{1 - \hat{\mu}_i}|_{x_i=1}}{\frac{\hat{\mu}_i}{1 - \hat{\mu}_i}|_{x_i=0}}
#' = \frac{\exp(\hat{\beta}_0 + 1 \times \hat{\beta}_1)}{\exp(\hat{\beta}_0 + 0 \times \hat{\beta}_1)}
#' = \exp(\hat{\beta}_1)
#' $$
#' 
#' - $\beta_1$ is log odds ratio: 
#'     - The log odds of being $y=1$ than $y=0$ will increase by $\beta_1$ if moving from $x=0$ to $x=1$
#' - $\exp(\beta_1)$ is odds ratio: 
#'     - The odds of being $y=1$ than $y=0$ will increase by $\exp(\beta_1)$ if moving from $x=0$ to $x=1$
#' 
#' 
#' interpret logistic regression coefficient $\beta$ adjusting for other covaraites
#' ===
#' 
#' - $logit(\mu_i) = \log(\frac{\mu_i}{1 - \mu_i}) = \beta_0 + \beta_1 x_i + Z_i^\top\gamma$
#' - for predictor x
#' 
#' $$\hat{OR} = \frac{\hat{odds}|_{x_i=1}}{\hat{odds}|_{x_i=0}}
#' = \frac{\frac{\hat{\mu}_i}{1 - \hat{\mu}_i}|_{x_i=1}}{\frac{\hat{\mu}_i}{1 - \hat{\mu}_i}|_{x_i=0}}
#' = \frac{\exp(\hat{\beta}_0 + 1 \times \hat{\beta}_1 + Z_i^\top\gamma)}{\exp(\hat{\beta}_0 + 0 \times \hat{\beta}_1 + Z_i^\top\gamma)}
#' = \exp(\hat{\beta}_1)
#' $$
#' 
#' - $\beta_1$ is log odds ratio:
#'     - The log odds of being $y=1$ than $y=0$ will increase by $\beta_1$ if moving from $x=0$ to $x=1$, after adjusting for other covariates.
#' - $\exp(\beta_1)$ is odds ratio:
#'     - The odds of being $y=1$ than $y=0$ will increase by $\exp(\beta_1)$ if moving from $x=0$ to $x=1$, after adjusting for other covariates
#' 
#' 
#' 
#' logistic regression in R
#' ===
#' 
## ------------------------------------------------------------------------
glm_binomial3 <- glm(svi ~ lcavol + lweight + age, data = prostate,  family = binomial())
summary(glm_binomial3) # display results

#' 
#' ---
#' 
## ------------------------------------------------------------------------
confint(glm_binomial3) # 95% CI for the coefficients
exp(coef(glm_binomial3)) # exponentiated coefficients
exp(confint(glm_binomial3)) # 95% CI for exponentiated coefficients

#' 
#' Model selection for logistic regression (also all other GLM)
#' ===
#' 
## ------------------------------------------------------------------------
step(glm(svi ~ ., data = prostate,  family = binomial()))

#' 
#' Estimation for $\beta$
#' ===
#' 
#' - The estimating equation can be least square loss or likelihood function.
#' - Two methods are typically used to solve the estimating equations of $\beta$:
#'     - Newton-Raphson method (an iterative method for solving nonlinear equations)
#'     - Fisher scoring method (similar to the Newton-Raphson method but using the expected information or Hessian matrix instead of the observed information)
#' - Will learn how to solve this problem in the optimization lecture.
#' 
#' Multinomial logistic Regression 
#' ===
#' 
#' When the outcome categories $1,2,\ldots,J$ are unordered,
#' a multinomial logit model defines $J - 1$ logits with the form (treating $j=1$ as the reference category):
#' 
#' $$\log (odds_{j1}) = \log(\frac{\mu_{ij}}{\mu_{i1}}) = \log(\frac{P(Y_i = j)}{P(Y_i = 1)}) = \beta_{0j} + \sum_k \beta_{kj} X_{ik}$$
#' 
#' - For each model, there will be $J - 1$ predicted log odds for each category relative to the reference category.
#' - Interpretation of the coefficients are the same as that in a logistic regression but $J - 1$ intercepts and $J - 1$ sets of coefficients instead of one.
#' - The coefficient $\beta_{kj}$ can be interpreted as: the increase in log-odds of falling into category $j$ versus category 1 (the reference category) resulting from a one-unit increase in covariate $X_k$ holding the other covariates constant.
#' - When $J=2$, the multinomial logistic regression reduces to logistic regression.
#' 
#' Multinomial logistic Regression example
#' ===
#' 
#' - The program choices are general program, vocational program and academic program. 
#' - Their choices can be modeled using their writing score and their social economic status.
#' 
## ------------------------------------------------------------------------
ml <- read.csv("https://caleb-huo.github.io/teaching/data/mlogistic_program/hsbdemo.csv",row.names=1)
head(ml)

#' 
#' Multinomial logistic Regression result
#' ===
#' 
## ------------------------------------------------------------------------
ml$prog2 <- relevel(ml$prog, ref = "vocation")
ml$ses2 <- relevel(ml$ses, ref = "low")

library("nnet")
test <- multinom(prog2 ~ ses2 + write, data = ml)

summary(test)

#' 
#' 
#' Multinomial logistic Regression interpretation
#' ===
#' 
#' 1. A one-unit increase in write increases the log odds of being in academic program vs. vocation  program by 0.11
#' 2. A one-unit increase in write increases the log odds of being in general program vs. vocation program by 0.056
#' 3. The log odds of being in academic program than in vocation program will increases by 0.98 if moving from ses=”low” to ses=”high”.
#' 
#' 
#' ordinal logistic regression
#' ===
#' 
#' - An ordinal logistic regression model has $J-1$ logit functions
#' 
#' $$\log (\frac{P(Y > j | X)}{P(Y \le j | X)}) = \log (\frac{P(Y > j | X)}{1 - P(Y > j | X)}) = \alpha_j + \beta^T X$$
#' For all $j = 1, \ldots, J - 1$.
#' 
#' - When $J = 3$
#' 
#' $$\log (\frac{P(Y > 1 | X)}{P(Y \le 1 | X)}) = 
#' \log (\frac{P(Y = 2 | X) + P(Y = 3 | X)}{P(Y = 1 | X)}) = 
#' \alpha_1 + \beta^T X$$
#' 
#' $$\log (\frac{P(Y > 2 | X)}{P(Y \le 2 | X)}) = 
#' \log (\frac{P(Y = 3 | X)}{P(Y = 1 | X) + P(Y = 2 | X)}) = 
#' \alpha_2 + \beta^T X$$
#' 
#' 
#' 
#' ordinal logistic regression
#' ===
#' 
#' - An ordinal logistic regression model has $J-1$ logit functions
#' 
#' $$\log (\frac{P(Y > j | X)}{P(Y \le j | X)}) = \log (\frac{P(Y > j | X)}{1 - P(Y > j | X)}) = \alpha_j + \beta^T X$$
#' For all $j = 1, \ldots, J - 1$.
#' 
#' - The sign of the coefficient $\beta$ indicates that
#'     - If $\beta > 0$:  it is more likely to observe higher values of $Y$.
#'     - If $\beta < 0$:  it is more likely to observe lower values of $Y$.
#' 
#' Ordinal Logistic Regression example
#' ===
#' 
## ------------------------------------------------------------------------
dat <- read.csv("https://caleb-huo.github.io/teaching/data/ologit/ologit.csv")
head(dat)

#' 
#' Ordinal logistic Regression result
#' ===
#' 
## ------------------------------------------------------------------------
library(MASS)
dat$apply2 <- relevel(dat$apply, ref = "unlikely")
m <- polr(apply2 ~ pared + public + gpa, data = dat, Hess=TRUE)
summary(m)
exp(coef(m))

#' 
#' 
#' Ordinal logistic Regression interpretation
#' ===
#' 
#' 1. One unit increase in parental education, from 0 (Low) to 1 (High), the odds of “very likely” applying versus “somewhat likely” or “unlikely” applying combined are 2.85 greater .
#' 
#' 2. The odds “very likely” or “somewhat likely” applying versus “unlikely” applying is 2.85 times greater .
#' 
#' 3. For gpa, when a student’s gpa increases 1 unit, the odds of moving from “unlikely” applying to “somewhat likely” or “very likely” applying (or from the lower and middle categories to the high category) are multiplied by 1.85.
#' 
#' 
#' GLM for count data
#' ===
#' 
#' - Possion regression
#' - Negative bionomial
#' 
#' Poisson Model (Components of GLM for count data)
#' ===
#' 
#' - To analyze data where the outcome $Y$ is a count, we can use GLM.
#' - The random component is the Poisson distribution.
#'     - $Y|X \sim Poisson(\mu)$
#' - The systematic compoents is linear combination of the explanatory variables $X^\top \beta$
#' - The most common link used is the log link, which is the canonical link from exponential family.
#' - Poisson model for counts with log link has the form $\log (\mu) = X^\top \beta$, which also refers to Poisson loglinear model.
#' 
#' Poisson distribution
#' ===
#' 
#' 
#' - Let $\mu$ be the rate of occurrence of an event, or the expected number of times an event will occur during a given period.
#' - Let Y be a random variable indicating the number times the event did occur. If $Y \sim Poisson(\mu)$, then
#' $$P(Y = y;\mu) = \frac{\exp(-\mu)\mu^y}{y!},$$
#' - $y = 0,1,2, \ldots$
#' - $E(Y) = \mu = Var(Y)$
#' 
#' Poisson Model
#' ===
#' 
#' - For Poisson Model, can we use the identify link instead of the log link?
#' - This would be a linear model and give $E(Y|X) = \mu = X^\top \beta$
#' - The problem with this formulation is that it can yield values of $\mu < 0$
#' 
#' 
#' fit Poisson Model in R
#' ===
#' 
## ------------------------------------------------------------------------
library(pscl)
head(bioChemists)

#' 
#' 
#' fit Poisson Model in R
#' ===
## ------------------------------------------------------------------------
glm_poisson <- glm(art ~ . , data = bioChemists,  family = poisson())
summary(glm_poisson) # display results

#' 
#' ---
#' 
## ------------------------------------------------------------------------
confint(glm_poisson) # 95% CI for the coefficients
exp(coef(glm_poisson)) # exponentiated coefficients
exp(confint(glm_poisson)) # 95% CI for exponentiated coefficients

#' 
#' Interpretation of $\beta$ for Poisson model
#' ===
#' 
#' - Suppose there is only one covariate x
#' - $\log E(Y|X) = \log(\mu) = \beta_0 + \beta_1 X$
#'     - $X = 0$, $\log(\mu | X=0) = \beta_0$, $\mu_0 = \exp (\beta_0)$
#'     - $X = 1$, $\log(\mu | X=1) = \beta_0 + \beta_1$, $\mu_1 = \exp (\beta_0 + \beta_1)$
#' - $\log(\mu_1) - \log(\mu_0) = \beta_1$
#' - $\log(\frac{\mu_1}{\mu_0}) = \beta_1$
#' - $\mu_1 = \exp(\beta_1) \mu_0$
#' 
#' $\beta_1$ is difference in log of expected counts when $X$ increases by 1 unit.
#' 
#' $\exp (\beta_1)$ is multiplicative effect of the mean of $Y$ when $X$ increases by 1 unit.
#' 
#' 
#' Model selection for Poisson regression (also all other GLM)
#' ===
#' 
## ------------------------------------------------------------------------
step(glm_poisson)

#' 
#' 
#' 
#' 
#' Poission regression for Rate data
#' ===
#' 
#' - Events can occur over time and the length of time can vary from observation to observation.
#' - A rate is given by $\lambda_i = \mu_i/n_i = E(Y_i|X_i)/n_i$, where $n_i$ is follow-up time.
#' - We can write a Poisson model in terms of the rates using $\log(\lambda_i) = \log(\mu_i/n_i) = \log(\mu_i) - \log(n_i)$
#' - The following Poisson log-linear models for rates are equivalent:
#'     - $\log(\lambda_i) = X_i^\top \beta$
#'     - $\log(\mu_i) - \log(n_i) = X_i^\top \beta$
#'     - $\log(\mu_i) = \log(n_i) + X_i^\top \beta$
#' 
#' 
#' Modeling the rate
#' ===
#' 
#' - Poisson model for rate: $\log(\mu_i) = \log(n_i) + X_i^\top \beta$
#'     - $n_i$ is called the exposure.
#'     - $\log (n_i)$ is called the offset term
#' 
#' 
## ------------------------------------------------------------------------
burnData <- read.csv("https://caleb-huo.github.io/teaching/data/Burn/burn.csv", row.names = 1)
head(burnData)

#' 
#' Modeling the rate
#' ===
#' 
## ------------------------------------------------------------------------
glm_poisson_rate <- glm(infection ~ . - numfup + offset(log(numfup)), data = burnData,  family = poisson())
summary(glm_poisson_rate) # display results

#' 
#' Poisson Overdispersion
#' ===
#' 
#' - Poisson regression is the standard method used to model count (and rate) response data.
#' - Poisson distribution assumes the equality of its mean and variance, which is a property that is rarely found in real data.
#' - Response variance is greater than the mean in Poisson models is called **(Poisson) overdispersion**.
#' 
#' 
#' bioChemists data
#' ===
#' 
## ------------------------------------------------------------------------
mean(bioChemists$art)
var(bioChemists$art)

#' 
#' Negative Bionomial Regression
#' ===
#' 
#' - **Negative binomial (NB) regression** is a standard method to model overdispersed Poisson data.
#' - Negative binomial can be derived from a Poisson-gamma mixture model.
#'   - Let random variable $Y|\lambda \sim Poisson(\lambda)$ and $\lambda \sim gamma(\alpha, \beta)$.
#'   - The marginal distribution $Y$ follows a negative binomial distribution with $r = \alpha$ and $p = 1/(\beta + 1)$
#' 
#' $$Y|\lambda \sim Poisson(\lambda)$$
#' $$\lambda \sim gamma(\alpha, \beta)$$
#' $$Y \sim NB(\alpha, 1/(\beta + 1))$$
#' 
#' Negative Binomial Distribution
#' ===
#' 
#' - Negative binomial pdf (the number of failures before the $r$th success occurs)
#' $$f(y;r,p) = \binom{y + r - 1}{r - 1} p^r (1 - p)^y$$
#'     - $0 \le p \le 1$ and $r$ is a positive integer
#'     - $y + r$ is total number of trials.
#' 
#' - Exponential family form:
#' $$f(y;r,p) = \exp\{y\log(1 - p) + r\log p + \log \binom{y + r - 1}{r - 1} \}$$
#'     - Canonical parameter: $\theta = \log (1 - p) \rightarrow p = 1 - \exp(\theta)$
#'     - $b(\theta) = -r \log (p) = -r \log (1 - \exp(\theta))$
#'     - Scale parameter: $a(\phi) = 1$
#' 
#' The canonical link for the negative binomial distribution is rather complicated and hard to interpret, so it is rarely used. Instead to facilitate comparisons with the Poisson generalized linear model, a log link is typically used.
#' 
#' Negative Binomial Distribution
#' ===
#' 
#' - Negative bionomial mean: 
#' $$b'(\theta) = \frac{\partial b}{\partial p} \times \frac{\partial p}{\partial \theta}
#' = -\frac{r}{p}\{-(1-p)\} = \frac{r(1-p)}{p}
#' $$
#' - Negative bionomial variance: 
#' 
#' $$b''(\theta) = \frac{\partial^2 b}{\partial p^2} \times (\frac{\partial p}{\partial \theta})^2
#' + \frac{\partial b}{\partial p} \times \frac{\partial^2 p}{\partial \theta^2}
#' = \frac{r(1-p)}{p^2}
#' $$
#' 
#' - Reparametrization:
#'     - $\alpha = 1/r$
#'     - $\mu =  \frac{r(1-p)}{p}$
#' 
#' - Negative bionomial mean: 
#' $$b'(\theta) = \frac{\partial b}{\partial p} \times \frac{\partial p}{\partial \theta}
#' = \mu
#' $$
#' - Negative bionomial variance: 
#' 
#' $$b''(\theta) = \frac{\partial^2 b}{\partial p^2} \times (\frac{\partial p}{\partial \theta})^2
#' + \frac{\partial b}{\partial p} \times \frac{\partial^2 p}{\partial \theta^2}
#' = \mu + \alpha \mu^2
#' $$
#' The variance is always larger than the mean for the negative binomial.
#' 
#' NB is possion if $\alpha = 0$
#' 
#' Negative binomial example 1
#' ===
#' 
## ------------------------------------------------------------------------
library(MASS)
glm_nb <- glm.nb(art ~ . , data = bioChemists)
summary(glm_nb) # display results

#' 
#' Negative binomial example 2
#' ===
#' 
## ------------------------------------------------------------------------
glm_nb_rate <- glm.nb(infection ~ . - numfup + offset(log(numfup)), data = burnData)
summary(glm_nb_rate)

#' 
#' 
#' Interpretation Using the Count (or Rate)
#' ===
#' 
#' - As the mean structure for the negative binomial regression is identical to that for the Poisson regression, the same methods of interpretation based on $E(Y=y|z,x)$ can be used
#' - Model: $\log E(Y=y|Z,X) = \gamma Z + \beta X$
#' $$\frac{E(Y=y|Z=z,X=x+\Delta)}{E(Y=y|Z=z,X=x)} = \exp(\beta \Delta)$$
#' 
#' For a change of $\Delta$ in $x$, the expected count increases by a factor of $\exp(\gamma \Delta)$ holding other variables $Z = z$ constant.
#' 
#' Other popular GLM for count data
#' ===
#' 
#' - zero inflated count data (excessive zeros)
#'     - zero inflated Poisson regressoin
#'     - zero inflated negative binomial regressoin
#' - zero truncated count data: (zero cannot occur)
#'     - zero truncated Poisson regressoin
#'     - zero truncated negative binomial regressoin
#' 
#' 
#' zero inflated Poisson regression
#' ===
#' 
#' - The zero-inflated Poisson (ZIP) regression is used for count data with excess zeros.
#' - The data distribution combines the Poisson distribution and the Bernoulli distribution.
#' 
#' 
#' $$
#' P(y_i = j)=
#' \begin{cases}
#' \pi_i + (1 - \pi_i) \exp (-\mu_i), \mbox{if } j=0\\
#' (1 - \pi_i) \frac{\mu_i^{y_i} \exp(-\mu_i)}{y_i!}, \mbox{if } j>0
#' \end{cases}
#' $$
#' - The Poisson component is
#' $$\log(\mu_i) = \beta_0x_{10} + \beta_1 x_{1i} + \ldots + \beta_px_{1p}$$
#' - The zero proportion component is
#' $$\pi_i = \frac{\exp(\gamma_0z_{10} + \gamma_1 z_{1i} + \ldots + \gamma_pz_{1p})}{1 + \exp(\gamma_0z_{10} + \gamma_1 z_{1i} + \ldots + \gamma_pz_{1p})}$$
#' 
#' ZIP in R
#' ===
#' 
#' - zeroinfl:
#'   - y ~ x | z
#'   - x: for the Poisson component
#'   - z: for the zero proportion component
#' 
#' ```
#' fm_pois <- glm(art ~ ., data = bioChemists, family = poisson) 
#' ## without inflation
#' 
#' fm_zip <- zeroinfl(art ~ . | 1, data = bioChemists) 
#' ## with simple inflation (no regressors for zero component)
#' 
#' fm_zip2 <- zeroinfl(art ~ . | ., data = bioChemists) 
#' ## inflation with regressors
#' ## ("art ~ . | ." is "art ~ fem + mar + kid5 + phd + ment | fem + mar + kid5 + phd + ment")
#' ```
#' 
#' 
#' ZINB in R
#' ===
#' 
#' - The zero-inflated negative binomial (ZINB) regression is used for count data with overdispersion and excess zeros.
#' 
#' 
#' ```
#' fm_nb <- MASS::glm.nb(art ~ ., data = bioChemists)
#' ## without inflation
#' 
#' fm_zinb <- zeroinfl(art ~ . | 1, data = bioChemists, dist = "negbin")
#' ## with simple inflation (no regressors for zero component)
#' 
#' fm_zinb2 <- zeroinfl(art ~ . | ., data = bioChemists, dist = "negbin")
#' ## inflation with regressors
#' ## ("art ~ . | ." is "art ~ fem + mar + kid5 + phd + ment | fem + mar + kid5 + phd + ment")
#' ```
#' 
#' 
#' 
#' Further extensions
#' ===
#' We can extend these conditions for linear regression model
#' 
#' - Random (stochastic) component
#'     - $\mu = E(Y|X), Y|X \sim f(\mu)$, $f$ is a distribution from the exponential family.
#' - Systematic component 
#'     – Assume linear model: $X^\top\beta$
#' - Link function
#'     - $X^\top\beta = g(\mu)$
#' 
#' 
#' How about non linear systematic component:
#' 
#' - generalized additive model
#'     - Assume additive models for the link function: $g(\mu) = \beta_0 + \sum_p r_p (x_p)$
#' 
