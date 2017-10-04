#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Wednesday October 4th, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Ridge regression, Lasso and elastic net.
#' ---
#' 
#' Outlines
#' ===
#' - Ridge regression
#' - Lasso
#' - Elastic net
#' 
#' linear regression model
#' ===
#' - $Y = (Y_1, \ldots, Y_n)^\top \in \mathbb{R}^n$
#' - $X = (1_n, X_1, \ldots,  X_p) \in \mathbb{R}^{n \times (p+1)}$
#'     - $1_n \in \mathbb{R}^n$, 
#'     - $X_j \in \mathbb{R}^n$ with $1 \le j \le p$
#' - $\beta = (\beta_0, \beta_1, \ldots, \beta_p)^\top \in \mathbb{R}^{p+1}$
#' - $\varepsilon = (\varepsilon_1, \ldots, \varepsilon_n)^\top \in \mathbb{R}^n$
#' 
#' $$Y = X\beta + \varepsilon$$
#' 
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
#' Problem with linear regression
#' ===
#' 
#' - When $n>p$, number of subjects is larger than number of features (variables), linear model works fine.
#' 
#' - When $n<p$, $\hat{\beta} = (X^\top X)^{-1} X^\top Y$, $X^\top X \in \mathbb{R}^{p \times p}$ is singular.
#' Thus we cannot estimate $\hat{\beta}$
#'     - Solution: apply PCA to $X$ and reduce to $X' \in \mathbb{R}^{n \times r}$, where $r<n$. (interpretation)
#'     - model selection: using backward selection, forward selection with BIC or AIC. (Searching space is large)
#'     
#' - When $n>p$, if two features are highly correlated, the resulting coefficients will have high variance and thus not stable.
#'     - Solution: PCA
#'     - Solution: use one one variable among all highly correlated variables as representative (using VIF to detect)
#' 
#' Collinearity
#' ===
#' 
## ------------------------------------------------------------------------
n <- 100
set.seed(32611)
x1 <- rnorm(n,3)
x2 <- rnorm(n,5)
x3 <- x2 + rnorm(n,sd=0.1)
cor(x2, x3)
x <- data.frame(x1,x2,x3)
y <- 2*x1 + 3*x2 + 4*x3 + rnorm(n, sd = 3)
xyData <- cbind(y,x)
lmFit <- lm(y~x1 + x2 + x3, data=xyData)
summary(lmFit)

#' 
#' How to test collinearity
#' ===
#' 
#' - Remove variable with VIF > 10
#' 
## ------------------------------------------------------------------------
library(car)
vif(lmFit)

summary(lm(y~x1 + x2, data=xyData))

#' 
#' 
#' To solve these problems
#' ===
#' In the past several decades, regularization methods provide better solutions to this problem
#' 
#' - ridge regression.
#' - lasso.
#' - Elastic net
#' 
#' 
#' Ridge regression
#' ===
#' 
#' $$\hat{\beta} = \arg \min_\beta \frac{1}{2}\| Y - X\beta\|_2^2 + \lambda \| \beta\|^2_2$$
#'     - $\|a\|_2 = \sqrt{a_1^2 + \ldots + a_p^2}$.
#'     
#' - $\lambda \ge 0$ is a tuning parameter, controling the strength of the penalty term.
#'     - $\lambda =0$, we have original linear regression $\hat{\beta}^{ridge} = \hat{\beta}^{LS}$.
#'     - $\lambda = \infty$, we get $\hat{\beta}^{ridge} = 0$
#'     - For $\lambda$, we both fit a linear model and shrink the coefficients.
#' 
#' 
#' Ridge regression solution
#' ===
#' 
#' $$\hat{\beta} = \arg \min_\beta \frac{1}{2}\| Y - X\beta\|_2^2 + \lambda \| \beta\|^2_2$$
#' 
#' - $\hat{\beta} = (X^\top X + \lambda I)^{-1} X^\top Y$
#' - As $\lambda$ increases, the bias increases and the variance decreases.
#' 
## ------------------------------------------------------------------------
library(MASS)
lm.ridge(y~x1 + x2 + x3, data=xyData, lambda = 10)

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
#' 
#' 
#' ridge regression Prostate cancer data example
#' ===
#' 
## ------------------------------------------------------------------------
library(MASS)
lm_ridge <- lm.ridge(lpsa ~ ., data=prostate, lambda=0); lm_ridge
lm(lpsa ~ ., data=prostate)

#' 
#' ridge regression Prostate cancer data example 2
#' ===
#' 
## ------------------------------------------------------------------------
lm.ridge(lpsa ~ ., data=prostate, lambda=10)
lm.ridge(lpsa ~ ., data=prostate, lambda=Inf)

#' 
#' 
#' Summary for Ridge regression
#' ===
#' 
#' - Ridge regression will stabilize the varaince of the coefficient estiamtes.
#' - Ridge regression will increase bias but decrease variance.
#' - Problem with ridge regression: coefficient won't be shrinked to exact 0.
#' 
#' lasso
#' ===
#' 
#' - The full name of Lasso is least absolute shrinkage and selection operator (Tibshirani, 1996)
#' - Lasso, the $l_1$ norm penalty will shrink some of the coefficients to exact zero.
#' $$\hat{\beta}^{lasso} = \arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1$$
#' - $\|\beta\|_1 = \sum_{j=1}^p |\beta_j|$ is called $l_1$ norm of $\beta$.
#' - $\lambda \ge 0$ is a tuning parameter, controling the strength of the penalty term.
#'     - $\lambda =0$, we have original linear regression.
#'     - $\lambda = \infty$, we get $\hat{\beta}^{lasso} = 0$
#'     - For $\lambda$, we both fit a linear model and shrink some coefficients to **exact zero**.
#' 
#' lasso example
#' ===
#' - lasso was implemented in R lars package.
## ------------------------------------------------------------------------
library(lars)
x <- as.matrix(prostate[,1:8])
y <- prostate[,9]
lassoFit <- lars(x, y) ## lar for least angle regression
coef(lassoFit, s=2, mode="lambda") ## get beta estimate when lambda = 2

#' 
#' Re-visit AIC and BIC
#' ===
#' 
#' - AIC and BIC will achieve feature selection.
#' - equivalently:
#' $$\hat{\beta}^{lasso} = \arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2 + \lambda \|\beta\|_0$$
#' 
#' $\|\beta\|_0$ equals to $k$ where $k$ is number of non-zero entries of $\beta$.
#' 
#' visualize lasso path
#' ===
#' - lasso solution (beta estimate is piecewise linear with respect to lambda)
## ------------------------------------------------------------------------
plot(lassoFit)

#' 
#' 
#' Intuition for lasso and ridge regression
#' ===
#' - lasso regression equivalent forms.
#'     - in penalty form:
#' $$\hat{\beta}^{lasso} = \arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2 + \lambda  \|\beta\|_1$$
#'     - in constraint form:
#' $$\hat{\beta}^{lasso} = \arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2, s.t,  \|\beta\|_1 \le \mu$$ 
#' 
#' 
#' - ridge regression equivalent forms.
#'     - in penalty form:
#' $$\hat{\beta}^{ridge} = \arg\min_{\beta\in\mathbb{R}^p}
#' \|y - X\beta\|_2^2 + \lambda\|\beta\|_2^2$$
#'     - in constraint form:
#' $$\hat{\beta}^{ridge} = \arg\min_{\beta\in\mathbb{R}^p}
#' \|y - X\beta\|_2^2, s.t, \|\beta\|_2^2 \le \mu$$
#' 
#' Intuition for lasso and ridge regression
#' ===
#' - Lasso: $\hat{\beta}^{lasso} = \arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2, s.t,  \|\beta\|_1 \le \mu$
#' - Ridge: $\hat{\beta}^{ridge} = \arg\min_{\beta\in\mathbb{R}^p}
#' \|y - X\beta\|_2^2, s.t, \|\beta\|_2^2 \le \mu$
#' 
#' ![From book Element of Statistical Learning](../figure/lasso.png)
#' 
#' How to choose Tuning parameter
#' ===
#' - cross validation, leave this for future lecture.
#' 
#' lasso path
#' ===
#' 
## ------------------------------------------------------------------------
plot(lassoFit)

#' 
#' - lasso solution is piecewise linear.
#' - we only need to calculate the solution at the knots
#' 
#' 
#' 
#' Elastic net
#' ===
#' 
#' $$\hat{\beta}^{elastic} = \arg\min_{\beta \in \mathbb{R}^P} \|y - X\beta\|_2^2 + \lambda_2  \|\beta\|_2 + \lambda_1  \|\beta\|_1$$
#' 
#' ![](http://scikit-learn.sourceforge.net/0.7/_images/plot_sgd_penalties1.png)
#' 
#' - In their paper, they claim elastic net has smaller Mean squared error.
#' 
#' Group lasso
#' ===
#' 
#' $$\min_{\beta=(\beta_{(1)},\dots,\beta_{(G)}) \in \mathbb{R}^p} \frac{1}{2} ||y-X\beta||_2^2 + \lambda \sum_{i=1}^G \sqrt{p_{(i)}} ||\beta_{(i)}||_2,$$
#' 
#' - Where $y \in \mathbb{R}^n$ is outcome and $X \in \mathbb{R}^{n\times p}$ is the design matrix.
#' - The design matrix can be partitioned in to $G$ groups $X = [X_{(1)}, \ldots, X_{(G)}]$ where $X_{(i)} \in \mathbb{R}^{n \times p_{(i)}}$.
#' - $\beta \in \mathbb{R}^p$ is the predictor and it is partitioned in to $G$ groups.
#' 
#' ![](../figure/groupLasso.png)
#' 
#' Fused lasso
#' ===
#' 
#' $$\min_{\beta \in \mathbb{R}^p} \frac{1}{2} || y - \beta ||_2^2 + \lambda \sum_{i=1}^{p-1} |\beta_i - \beta_{i+1}|$$
#' 
#' ![](http://statweb.stanford.edu/~bjk/regreg/_images/fusedlassoapprox.png)
#' 
#' 
#' Generalized lasso
#' ===
#' 
#' Consider a general setting
#' $$\min_{\beta \in \mathbb{R}^p} f(\beta) + \lambda ||D\beta||_1$$
#' where $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is a smooth convex function. 
#' $D \in \mathbb{R}^{m\times n}$ is a penalty matrix.
#' When $D=I$, the formulation will reduce to the lasso regression problem.
#' 
#' When $$D= \left( \begin{array}{cccccc}
#' -1 & 1 & 0 & \ldots & 0 & 0 \\
#' 1 & -1 & 1 & \ldots & 0 & 0 \\
#' \vdots  & \vdots  & \vdots  & \ddots & \vdots & \vdots \\
#'  0 & 0 & 0 & \ldots & -1 & 1
#'  \end{array} \right) $$,
#'  The penalty will be the fussed lasso penalty.
#'  
#' Last slide
#' ===
## ------------------------------------------------------------------------
knitr::purl("lasso.Rmd", output = "lasso.R ", documentation = 2)

