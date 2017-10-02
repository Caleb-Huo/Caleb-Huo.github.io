#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday October 2, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Linear Model
#' ---
#' 
#' Outlines
#' ===
#' - Linear regression
#' - model diagnositic
#' - model selection
#' - spline
#' 
#' 
#' linear regression model (population version)
#' ===
#' 
#' - $Y \in \mathbb{R}$, $X_j \in \mathbb{R}$ with $j = \{1, \ldots, p\}$, $\beta_j \in \mathbb{R}$ with $j = \{0, \ldots, p\}$, $\varepsilon \in \mathbb{R}$
#' 
#' $$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_pX_p + \varepsilon, \text{with }\varepsilon \sim N(0, \sigma^2)$$
#' 
#' - Y: outcome variable or response.
#' - $X_1, X_2, \ldots, X_p$ are called predictors.
#' - $\beta_0, \beta_1, \ldots, \beta_p$ are called coefficient, for which we will estimate.
#' 
#' Assumptions:
#' 
#' - $Y$ is a linear combination of $X_1, X_2, \ldots, X_p$.
#' - $\varepsilon \sim N(0, \sigma^2)$.
#' 
#' linear regression model (sample version)
#' ===
#' 
#' - $Y = (Y_1, \ldots, Y_n)^\top \in \mathbb{R}^n$
#' - $X = (1_n, X_1, \ldots,  X_p) \in \mathbb{R}^{n \times (p+1)}$
#'     - $1_n \in \mathbb{R}^n$, 
#'     - $X_j \in \mathbb{R}^n$ with $1 \le j \le p$
#' - $\beta = (\beta_0, \beta_1, \ldots, \beta_p)^\top \in \mathbb{R}^{p+1}$
#' - $\varepsilon = (\varepsilon_1, \ldots, \varepsilon_n)^\top \in \mathbb{R}^n$
#' 
#' $$Y = X\beta + \varepsilon$$
#' 
#' Assumptions:
#' 
#' - $Y$ is a linear combination of $1_n, X_1, X_2, \ldots, X_p$.
#' - $\varepsilon_i \sim N(0, \sigma^2)$.
#' - $\varepsilon_i$'s are independently identically distributed (iid).
#' 
#' 
#' Solution to linear regression model
#' ===
#' 
#' - least square estimator (LS):
#' $$\hat{\beta} = \arg \min_\beta \frac{1}{2}\| Y - X\beta\|_2^2$$
#'     - $\|a\|_2 = \sqrt{a_1^2 + \ldots + a_p^2}$.
#' 
#' - Maximum likelihood estimator (MLE):
#' $$\hat{\beta} = \arg \max_\beta L(\beta; X, Y)$$
#' 
#' - For linear regression model, LS estimator is the same as MLE
#' $$\hat{\beta} = (X^\top X)^{-1} X^\top Y$$
#' Assuming $X^\top X$ is invertable
#' 
#' Fit linear model in R, lm()
#' ===
#' 
#' Basic syntax for lm()
## ---- eval=F-------------------------------------------------------------
## lm(formula, data)

#' 
#' - formula: Symbolic description of the model
#' - data: Optional dataframe containing the variables in the model
#' 
#' - summary() function summarize the linear model result
#' - plot() function output model diagnostic results
#' 
## ------------------------------------------------------------------------
lmfit <- lm(mpg ~ cyl, data=mtcars)
lmfit

#' 
#' ---
#' 
## ------------------------------------------------------------------------
summary(lmfit)

#' 
#' ---
#' 
## ------------------------------------------------------------------------
plot(lmfit)

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
#' ---
#' 
#' Explanation of these variables:
#' 
#' - lpsa: log PSA score
#' - lcavol: log cancer volume
#' - lweight: log prostate weight
#' - age: age
#' - lbph: log of the amount of benign prostatic hyperplasia
#' - svi: seminal vesicle invasion
#' - lcp: log of capsular penetration
#' - gleason: Gleason score
#' - pgg45: percent of Gleason scores 4 or 5.
#' - train: a logical vector indicating training data or testing data (ignore at this moment)
#' 
#' ---
#' 
## ------------------------------------------------------------------------
head(prostate)

#' 
#' Exploratory analysis
#' ===
#' - Distribution of each varaible.
#' - correlation between each pair of vairables.
#' - visualizing the scattered plot of each pair of variables.
#' 
#' 
#' Distribution of each varaible -- Histogram
#' ===
## ------------------------------------------------------------------------
par(mfrow=c(3,3))
for(i in 1:9){
  aname <- names(prostate)[i]
  hist(prostate[[i]], main=aname, col="steelblue", breaks=20)
}

#' 
#' ---
#' 
#' What can we learn from the histogram.
#' 
#' - lcp, the log amount of capsular penetration, is very skewed, a bunch of men with little, then a big spread
#' - svi (the presence of seminal vesicle invasion) is binary
#' - gleason score, takes integer values of 6 and larger; 
#' - lspa, the log PSA score, is likely to be normally distributed
#' 
#' Correlation between each pairs of variables 
#' ===
## ------------------------------------------------------------------------
cor_prostate <- cor(prostate)
max(cor_prostate[cor_prostate!=1])
maxID <- arrayInd(which(max(cor_prostate[cor_prostate!=1]) == cor_prostate), dim(cor_prostate))
names(prostate[maxID[1,]])


#' 
#' ---
#' 
## ------------------------------------------------------------------------
heatmap(cor_prostate, scale="none") ## instead of image. Rowv = F, Colv = F to suppress hierarchical clusteirng dendrogram

#' 
#' visualizing the scattered plot of each pair of variables.
#' ===
## ------------------------------------------------------------------------
pairs(prostate,pch=19)

#' 
#' Formulas
#' ===
#' 
#' 
#' - Basic form of a formula:
#'     - response ~ model
#' - Formula notation:
#' 
#' Syntax  | function
#' ------- | -----------------------------------------------------------------
#' +       | Separate main effects
#' :       | Denote interactions
#' *       | main effects and interactions
#' ˆn      | include all main effects and n-order interactions
#' -       | remove the specified terms
#' I()     | Brackets the portions of a formula where operators are used mathematically
#' .       | Main effect for each column in the dataframe, except the response
#' 
#' 
#' 
#' 
#' Fit linear regression model
#' ===
#' 
#' - lpsa as indepedent variable/outcome variable
#' - age and lweight as dependent variable/predicvors
#' 
## ------------------------------------------------------------------------
lm_prostate <- lm(lpsa ~ age + lweight, data = prostate)
lm_prostate
class(lm_prostate)
names(lm_prostate)

#' 
#' Inference for Linear Models
#' ===
#' 
#' function  | application
#' ------- | -----------------------------------------------------------------
#' summary() | summary of linear model
#' coef()    | get coefficient
#' confint() | Confidence intervals for model parameters
#' predict() | make a prediction
#' fitted()  | get fitted value
#' 
#' ---
#' 
## ------------------------------------------------------------------------
summary(lm_prostate)

#' 
#' ---
#' 
## ------------------------------------------------------------------------
coef(lm_prostate)
confint(lm_prostate)

#' 
#' ---
#' 
#' 
## ------------------------------------------------------------------------
anew <- data.frame(age = 29, lweight = 100)
predict(lm_prostate, anew)

head(fitted(lm_prostate))

#' 
#' Formulas (continue)
#' ===
#' 
#' - Sample formulas, for a model with response y and predictors a, b and c
#'     
#' Model   | Interpretation
#' --------------------- | -----------------------------------------------------
#' y ~ 1   | Just the intercept
#' y ~ a   | One main effect
#' y ~ - 1 + a | No intercept
#' y ~ a+b+c+a:b   | Three main effects and an interaction between a and b
#' y ~ a * b       | All main effects and interactions (same as a+b+a:b)
#' y ~ factor(a)   | Create dummy variables for a (if not already a factor)
#' y ~ (a+b+c)ˆ2   | All main effects and second-order interactions
#' y ~ (a+b+c)ˆ2 - a:b   | All main effects and second-order interactions except a:b
#' y ~ I(aˆ2)            | Transform a to aˆ2
#' log(y) ~ a            | Transform a to aˆ2
#' y ~ a/b/c             | Factor c nested within factor b within factor a
#' y ~ .                 | Main effect for each column in the dataframe
#' 
#' 
#' Demonstrate other formula of lm using prostate cancer data
#' ===
#' 
## ---- eval=F-------------------------------------------------------------
## lm(lpsa ~ 1, data = prostate)
## lm(lpsa ~ age, data = prostate)
## lm(lpsa ~ - 1 + age, data = prostate)
## lm(lpsa ~ age + lweight + gleason + gleason:lweight, data = prostate)
## lm(lpsa ~ gleason*lweight, data = prostate)
## lm(lpsa ~ factor(gleason), data = prostate)
## lm(lpsa ~ (age + lweight + gleason)^2, data = prostate)
## lm(lpsa ~ (age + lweight + gleason)^2 - gleason:lweight, data = prostate)
## lm(lpsa ~ I(age^2), data = prostate)
## lm(log(age) ~ lpsa, data = prostate)
## lm(lpsa ~ ., data = prostate)
## lm(lpsa ~ as.factor(svi)/as.factor(gleason), data = prostate)

#' 
#' Anova (1)
#' ===
#' 
## ------------------------------------------------------------------------
blm <- lm(lpsa ~ as.factor(gleason), data = prostate)
summary(blm)
summary.aov(blm)

#' 
#' 
#' Anova (2)
#' ===
#' 
## ------------------------------------------------------------------------
clm <- aov(lpsa ~ as.factor(gleason), data = prostate)
summary(clm)
TukeyHSD(clm) ##Multiple comparisons, Tukey’s Honest Significant Difference

#' 
#' ANCOVA
#' ===
#' 
## ------------------------------------------------------------------------
dlm <- aov(lpsa ~ as.factor(gleason) + age, data = prostate)
summary(dlm)

#' 
#' Model Diagnostics
#' ===
#' 
#' function  | application
#' ------------------ | -----------------------------------------------------------------
#' fitted.values() | Returns fitted values
#' residuals()    | Returns residuals
#' rstandard() | Standardized residuals, variance one; residual standardized using overall error variance
#' rstudent() | Studentized residuals, variance one; residual standardized using leave-one-out measure of the error variance
#' qqnorm()  | Normal quantile plot
#' qqline()      | Add a line to the normal quantile plot
#' plot()      | Given a lm object produces six diagnostic plots, selected using the which argument; default is plots 1-3 and 5
#' 
#' plot() on lm object options
#' ===
#' 
#' 1. Residual versus fitted values
#' 2. Normal quantile-quantile plot
#' 3. sqrt(Standardized residuals) versus fitted values
#' 4. Cook's distance versus row labels
#' 5. Standardized residuals versus leverage along with contours of Cook’s distance
#' 6. Cook’s distance versus leverage/(1-leverage) with sqrt(Standardized residuals) contours
#' 
#' 
#' fitted value and residuals
#' ===
#' 
## ------------------------------------------------------------------------
lm_prostate <- lm(lpsa ~ age + lweight, data = prostate)

#' 
#' - fitted value: 
#' $$X\hat{\beta}$$
## ------------------------------------------------------------------------
head(fitted(lm_prostate))

#' 
#' - residuals: 
#' $$Y - X\hat{\beta}$$
## ------------------------------------------------------------------------
head(residuals(lm_prostate))

#' 
#' Assess Gaussian assumption.
#' ===
#' 
#' - We can use residuals to approximate errors.
#' - Therefore we can check if
#'     - errors are normally distributed
#'     - When normally distributed, if common standard deviation across fitted value
#'     
## ------------------------------------------------------------------------
plot(lm_prostate, 1)

#' 
#' QQ-plot
#' ===
#' 
#' - Q-Q plot is what's called a quantile-quantile plot.
#' - Used to check if a list of values if normally distributed.
#'     - the list of values are sorted from smallest to largest.
#'     - x axis are quantiles (qnorm) of empirical CDF.
#'     - y axis are the sorted value.
#' - A straight line for QQ-plot shows the list of values are normally distributed.
#' 
#' QQ-plot 1
#' ===
#' 
## ------------------------------------------------------------------------
set.seed(32611)
qqnorm(rnorm(1000))

#' 
#' 
#' QQ-plot 2
#' ===
#' 
## ------------------------------------------------------------------------
set.seed(32611)
qqnorm(rnorm(1000, sd=10))

#' 
#' 
#' QQ-plot 3
#' ===
#' 
## ------------------------------------------------------------------------
set.seed(32611)
qqnorm(1:1000)

#' 
#' 
#' QQ-plot 4
#' ===
#' 
## ------------------------------------------------------------------------
set.seed(32611)
qqnorm(rexp(1000))

#' 
#' Prostate cancer example
#' ===
## ------------------------------------------------------------------------
lm_prostate <- lm(lpsa ~ age + lweight, data = prostate)
plot(lm_prostate, 2)

#' 
#' 
#' Formal normality test
#' ===
#' 
## ------------------------------------------------------------------------
set.seed(32611)
shapiro.test(rnorm(1000))
shapiro.test(1:1000)
shapiro.test(residuals(lm_prostate))

#' 
#' 
#' Unusual and Influential Data
#' ===
#' 
#' - Outliers: An observation with large residual.
#'     - An outlier may indicate a sample peculiarity or may indicate a data entry error or other problem.
#' - Leverage: An observation with an extreme value on a predictor variable.
#'     - Leverage is a measure of how far an independent variable deviates from its mean.
#'     - These leverage points can have an effect on the estimate of regression coefficients.
#' - Influence: Influence can be thought of as the product of leverage and outlierness.
#'     - Removing the observation substantially changes the estimate of coefficients.
#' 
#' 
#' Outliers
#' ===
#' 
## ------------------------------------------------------------------------
plot(lm_prostate, 3)

#' 
#' 
#' Leverage
#' ===
#' 
## ------------------------------------------------------------------------
plot(lm_prostate, 5)

#' 
#' influential points: Cook's distance
#' ===
#' 
## ------------------------------------------------------------------------
plot(lm_prostate, 4)

#' 
#' 
#' 
#' Fit a linear regression model with all variables
#' ===
## ------------------------------------------------------------------------
lm_all <- lm(lpsa ~ ., data=prostate) ## . Rest of all the other vaiables.
summary(lm_all)

#' 
#' 
#' Model selection
#' ===
#' 
#' - AIC, BIC
#' - forward selection and backward selection
#' 
#' AIC, BIC
#' ===
#' - AIC: Akaike information criterion
#' - $AIC = 2k - 2 \log ( {\hat{L}} )$
#'     - k: number of parameters.
#'     - $\hat{L}$: likelihood given the coefficient estimates
## ------------------------------------------------------------------------
AIC(lm_all)

#' 
#' - BIC: Bayesian information criterion
#' - $BIC = log(n)k - 2 \log ( {\hat{L}} )$
## ------------------------------------------------------------------------
BIC(lm_all)

#' 
#' backward selection
#' ===
#' - step() function can perform forward selection, backward selection or both.
#'     - other direction options: c("both", "backward", "forward")
## ------------------------------------------------------------------------
step(lm(lpsa ~ ., data=prostate)) ## default is backward selection

#' 
#' --- 
#' - You may notice that AIC from step() and AIC() are different
#' - AIC/BIC can be different for different methods up to a constant.
## ------------------------------------------------------------------------
stepDetails <- step(lm(lpsa ~ ., data=prostate), trace=0)$anova
stepDetails$AIC[2] - stepDetails$AIC[1]

AIC(lm(lpsa ~ . - gleason, data=prostate)) - AIC(lm(lpsa ~ ., data=prostate))


#' 
#' 
#' non-linear predictor (1)
#' ===
#' 
## ------------------------------------------------------------------------
library(minpack.lm)

## simulate data from sin function
set.seed(32611)
n <- 100
x <- runif(n,0,2*pi)
y <- sin(x) + rnorm(n,sd=0.2)
plot(x,y)

#' 
#' fit $y = A \sin(x + B) + C$
#' ===
#' 
## ------------------------------------------------------------------------
getPred <- function(parS, xx) {	
		parS$A * sin(xx + parS$B) + parS$C
	}

residFun <- function(p, observed, xx) observed - getPred(p,xx)

parStart <- list(A=1, B=3, C=0)
nls.out <- nls.lm(par=parStart, fn = residFun, observed = y,	xx = x)
nls.out

aseq <- seq(0,2*pi,length.out = 1000)
plot(x,y)
with(nls.out$par, lines(aseq, A*sin(aseq + B) + C, col=2))

#' 
#' non-linear predictor (2)
#' ===
#' 
## ------------------------------------------------------------------------
set.seed(32611)
n <- 100; 
x <- runif(n,0,5)
y <- exp(x) + rnorm(n,sd=1)
plot(x,y)

#' 
#' fit $y = a \times \exp(bx)$ 
#' ===
#' 
## ------------------------------------------------------------------------
getPred <- function(parS, xx) {	
		parS$A * exp(parS$B * xx)
	}

residFun <- function(p, observed, xx) observed - getPred(p,xx)

parStart <- list(A=3, B=3)
nls.out <- nls.lm(par=parStart, fn = residFun, observed = y,	xx = x)
nls.out

aseq <- seq(0,5,length.out = 1000)
plot(x,y)
with(nls.out$par, lines(aseq, A*exp(B*aseq), col=2))

#' 
#' 
#' Splines
#' ===
#' 
#' - Non-parametric approach to fit the curve (usually $d=1$).
#' - These basis functions are splines.
#' - A kth order spline is a piecewise polynomial function of degree k, that is continuous and has continuous derivatives of orders $1, \ldots, k-1$, at its knot points.
#' 
#' Splines motivating example
#' ===
#' 
#' ![continuity at the knots, from book the "element of statistical leaning"](../figure/splines.png)
#' 
#' Splines basis
#' ===
#' 
#' How to parametrize the set of splines with knots at $t_1, \ldots, t_m$?
#' 
#' - truncated power basis:
#'     - $g_1(x) = 1$, $g_2(x) = x$, ..., $g_{k+1}(x) = x^k$
#'     - $g_{k+1+j}(x) = (x - t_j)^k_+$, $j = 1, \ldots, m$
#'     - $(\cdot)_+ =  \max(0, \cdot)$
#' 
#' - B-spline basis <https://en.wikipedia.org/wiki/B-spline>
#' 
#' 
#' Splines types
#' ===
#' 
#' - Regression splines
#' - Natural splines
#' - Smoothing splines
#' 
#' 
#' Regression splines
#' ===
#' 
#' - Given input $x_1, \ldots, x_n$ and output $y_1, \ldots, y_n$, we consider to fit function $f$ with $k^{th}$ order splines with given knots at $t_1, \ldots, t_m$
#' $$f(x) = \sum_{j=1}^{m+k+1} \beta_j g_j(x),$$
#' where $\beta_j, 1\le j \le m+k+1$ are coefficients and $g_j, 1\le j \le m+k+1$ are basis functions for order $k$ spline on the knots $t_1, \ldots, t_m$ (e.g. truncated power basis or B-spline basis).
#' 
#' - define $G \in \mathbb{R}^{n \times (m+k+1)}$, where $G_{ij} = g_j(x_i)$.
#' - The solution coefficients $\hat{\beta} = (\hat{\beta}_1, \ldots, \hat{\beta}_{m+k+1})$
#' 
#' $$\hat{\beta} = \arg \min_{\beta \in \mathbb{R}^{m+k+1}} \|y - G\beta\|_2^2$$
#' 
#' - We know the solution is $\hat{\beta} (G^\top G)^{-1} G^\top y$
#' 
#' Natural splines
#' ===
#' 
#' - The problem with regression splines is that the estimates have high variance at the boundaries of the input domain.
#' 
#' - A solution is to force the piecewise polynomial function have lower degree at the boundary.
#'     - $f$ is a polynomial of degree $k$ on each of $[t_1, t_2], \ldots, [t_{m-1}, t_m]$
#'     - $f$ is a polynomial of degree $(k-1)/2$ on each of $(-\infty, t_1]$ and $[t_m, \infty)$
#' 
#' 
#' Smoothing splines
#' ===
#' 
#' - placing knots at all inputs $x_1, \ldots, x_n$, circumvent knot selection problem.
#' 
#' $$\hat{f} = \arg\min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int_a^b (f^{((k+1)/2)})^2 dx$$
#'     - first term is least square loss
#'     - second term penalized if the devirative of $f$ is too wiggly.
#'     
#' - When $k = 3$, this is the cubic smoothing splines, which is very commonly used.
#' 
#' $$\hat{f} = \arg\min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \int_a^b (f''(x))^2 dx$$
#' - Remarkably, it happens that the minimizer of general smoothing spline is a natural $k^{th}$ order spline with knots at input points $x_1, \ldots, x_n$.
#' 
#' cubic smoothing spline in R (data points)
#' ===
#' 
#' - smooth.spline()
## ------------------------------------------------------------------------
plot(disp ~ mpg, data = mtcars, main = "data(mtcars)")

#' 
#' 
#' cubic smoothing spline in R
#' ===
#' 
## ------------------------------------------------------------------------
plot(disp ~ mpg, data = mtcars, main = "data(mtcars)  &  smoothing splines")
cars.spl <- with(mtcars, smooth.spline(mpg, disp))
cars.spl
lines(cars.spl, col = "blue")

#' 
#' generate R code
#' ===
## ------------------------------------------------------------------------
knitr::purl("linearModel.Rmd", output = "linearModel.R ", documentation = 2)

