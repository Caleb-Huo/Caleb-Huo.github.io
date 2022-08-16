#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday Nov 1, 2021"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Optimization
#' ---
#' 
#' 
#' Outline
#' ===
#' 
#' - Univariate Optimization
#'     - Optimization in R
#'     - Gradient decent
#'     - Newton's method
#' - Multivariate Optimization
#'     - Gradient decent
#'     - Newton's method
#'     - Coordinate decent
#' 
#' Optimization
#' ===
#' - In mathematics, "optimization" or "mathematical programming" refers to the selection of a best element (with regard to some criterion) from some set of available alternatives.
#' 
#' - A typical optimization problem usually consists of maximizing or minimizing a real function (objective function) by systematically choosing input values under certain constraint.
#' 
#' - "Convex programming" studies the case when the objective function is convex (minimization) or concave (maximization) and the constraint set is convex.
#' 
#' 
#' 
#' Convex function
#' ===
#' 
#' Convex function properties:
#' 
#' - Second derivative always greater than 0
#' - Tangent lines always underestimate the function
#' - $\forall t \in (0,1)$, $tf(a) + (1 - t)f(b) > f(ta + (1 - t)b)$
#' 
#' 
#' ![](../figure/convex2.png)
#' 
#' 
#' - Convex optimization always leads to a global minimizer
#' - Non-convex optimization usually leads to a local minimizer
#' 
#' Optimization functions in R
#' ===
#' 
#' - optimize(): One dimensional optimization, no gradient or Hessian 
#' - optim(): General purpose optimization, five possible methods, gradient optional 
#' - constrOptim(): Minimization of a function subject to linear inequality constraints, gradient optional 
#' - nlm(): Non-linear minimization, can optionally include the gradient and hessian of the function as attributes of the objective function 
#' - nlminb(): Minimization using PORT routines, can optionally include the gradient and Hessian of the objective function as additional arguments 
#' 
#' 
#' Univariate optimization example
#' ===
#' Suppose our objective function is 
#' $$f(x) = e^x + x^4$$
#' What is the minimum value of $f(x)$, what is the correponding $x$?
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) exp(x) + x^4
curve(f, from = -1, to = 1)

#' 
#' 
#' Use R optimize() function to perform the optimization
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) exp(x) + x^4
xmin <- optimize(f, interval = c(-10, 10))
xmin
curve(f, from = -1, to = 1)
points(xmin$minimum, xmin$objective, col="red", pch=19)

#' 
#' 
#' 
#' Gradient decent
#' ===
#' - Consider an unconstrained minimization of $f$, we want to find $x^*$ such that 
#' $$f(x^*) = \min_{x\in \mathbb{R}} f(x)$$
#' - Another way to formulate the problem is to find $x^*$ such that 
#' $$x^* = \arg_{x\in \mathbb{R}} \min f(x)$$
#' 
#' **Gradient decent**: choose initial $x^{(0)} \in \mathbb{R}$, repeat:
#' $$x^{(k)} = x^{(k - 1)} - t\times f'(x^{(k - 1)}), k = 1,2,3,\ldots$$
#' 
#' - Until $|x^{(k)} - x^{(k - 1)}| < \varepsilon$, (e.g. $\varepsilon = 10^{-6}$).
#' - $f'(x^{(k - 1)})$ is the gradient (derivative of $f$) evaluated at $x^{(k - 1)}$.
#' - $t$ is a step size for gradient decent procedure.
#' 
#' 
#' Interpretation
#' ===
#' - At each iteration $k$, consider Taylor expansion at $x^{(k - 1)}$:
#' $$f(y) = f(x^{(k - 1)}) + f'(x^{(k - 1)}) \times (y - x^{(k - 1)}) + \frac{1}{2} f''(x^{(k - 1)}) (y - x^{(k - 1)})^2 + \ldots$$
#' - Quadratic approximation:
#'     - Replacing $f''(x^{(k - 1)})$ with $\frac{1}{t}$
#'     - Ingore higher order terms
#' 
#' $$f(y) \approx f(x^{(k - 1)}) + f'(x^{(k - 1)}) \times (y - x^{(k - 1)}) + \frac{1}{2t}  (y - x^{(k - 1)})^2$$
#' 
#' - Minimizing the quadratic approximation.
#' $$x^{(k)} = \arg_y \min f(y)$$
#'     
#'     - Set $f'(x^{(k)}) = 0 \Leftrightarrow f'(x^{(k - 1)}) + \frac{1}{t}  (x^{(k)} - x^{(k - 1)}) = 0 \Leftrightarrow x^{(k)} = x^{(k - 1)} - t\times f'(x^{(k - 1)})$
#' 
#'     - This is exactly the same as the gradient decent procedure.
#' 
#' 
#' Visual interpretation
#' ===
#' 
#' ![](../figure/GDprocedure.png)
#' 
#' - iterate 1: $x^{(1)} = x^{(0)} - t\times f'(x^{(0)})$
#'     - quadratic approximation at <span style="color:red">$x_0$</span>, 
#'     - minimizer of the quadratic approximation is <span style="color:blue">$x_1$</span>
#' 
#' - iterate 2: $x^{(2)} = x^{(1)} - t\times f'(x^{(1)})$
#'     - quadratic approximation at <span style="color:blue">$x_1$</span>, 
#'     - minimizer of the quadratic approximation is <span style="color:green">$x_2$</span>
#' 
#' - ...
#' 
#' Gradient decent on our motivating example
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) exp(x) + x^4 ## original function
g <- function(x) exp(x) + 4 * x^3 ## gradient function
curve(f, from = -1, to = 1, lwd=2) ## visualize the objective function

x <- x0 <- 0.8; x_old <- 0; t <- 0.1; k <- 0; error <- 1e-6
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error){ ## there is a scale issue. 
  ## E.g., if x = 1e-7, x_old = 1e-6, relative change is large, absolute change is small
  ## We can monitor abs(x - x_old) / (abs(x) + abs(x_old))
  k <- k + 1
  x_old <- x
  x <- x_old - t*g(x_old)
  trace <- c(trace, x) ## collecting results
  points(x, f(x), col=k, pch=19)
}

trace_grad <- trace
print(trace)

#' 
#' 
#' 
#' A non-convex example
#' ===
#' ![](../figure/nonConvex.jpg)
#' 
#' - If the objective function has multiple local minimums.
#' - Gradient will still work, but may fall into a local minimum (might be global minimum).
#' - kmeans algorithm is a non-convex minimization problem.
#' 
#' 
#' Discussion, will gradient decent algorithm always converge?
#' ===
#' 
#' ![](../figure/GDprocedure.png)
#' 
#' 
#' No, see the example
#' ===
#' 
#' - Minimize $f(x) = x^2$
#' - Initial point $x_0 = 1$
#' - stepsize $t = 1.1$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) x^2 ## original function
g <- function(x) 2*x ## gradient function
curve(f, from = -10, to = 10, lwd=2) ## visualize the objective function

x <- x0 <- 1; x_old <- 0; t <- 1.1; k <- 0; error <- 1e-6
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error & k < 30){ ## there is a scale issue. 
  ## E.g., if x = 1e-7, x_old = 1e-6, relative change is large, absolute change is small
  ## We can monitor abs(x - x_old) / (abs(x) + abs(x_old))
  k <- k + 1
  x_old <- x
  x <- x_old - t*g(x_old)
  trace <- c(trace, x) ## collecting results
  points(x, f(x), col=k, pch=19)
  segments(x0=x, y0 = f(x), x1 = x_old, y1 = f(x_old))
}

print(trace)

#' 
#' 
#' Change to another stepsize
#' ===
#' 
#' - Minimize $f(x) = x^2$
#' - Initial point $x_0 = 1$
#' - stepsize $t = 0.2$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) x^2 ## original function
g <- function(x) 2*x ## gradient function
curve(f, from = -1, to = 1, lwd=2) ## visualize the objective function

x <- x0 <- 1; x_old <- 0; t <- 0.2; k <- 0; error <- 1e-6
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error & k < 30){
  k <- k + 1
  x_old <- x
  x <- x_old - t*g(x_old)
  trace <- c(trace, x) ## collecting results
  points(x, f(x), col=k, pch=19)
  segments(x0=x, y0 = f(x), x1 = x_old, y1 = f(x_old))
}

print(trace)

#' 
#' 
#' 
#' Change to another stepsize
#' ===
#' 
#' - Minimize $f(x) = x^2$
#' - Initial point $x_0 = 1$
#' - stepsize $t = 0.01$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) x^2 ## original function
g <- function(x) 2*x ## gradient function
curve(f, from = -1, to = 1, lwd=2) ## visualize the objective function

x <- x0 <- 1; x_old <- 0; t <- 0.01; k <- 0; error <- 1e-6
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error & k < 30){
  k <- k + 1
  x_old <- x
  x <- x_old - t*g(x_old)
  trace <- c(trace, x) ## collecting results
  points(x, f(x), col=k, pch=19)
  segments(x0=x, y0 = f(x), x1 = x_old, y1 = f(x_old))
}

print(trace)

#' 
#' 
#' 
#' How to select stepsize $t$, backtracking line search
#' ===
#' 
#' ![](../figure/backTracking.png)
#' 
#' 
#' How to select stepsize $t$, backtracking line search
#' ===
#' 
#' - Linear approximation (first order Taylor expansion) at the current point $x$.
#' $$f(x + t\Delta x) = f(x) + tf'(x)\Delta x$$
#' 
#' - Decrease the slope of the approximation by $\alpha$
#' 
#' $$f(x) + t\alpha f'(x)\Delta x$$
#' 
#' - A sufficient condition for a proposed $t$ to guarantee **decent** (decrease objective function):
#' $$f(x + t\Delta x) < f(x) + t\alpha f'(x)\Delta x$$
#' 
#' - Plugin $\Delta x = -f'(x)$
#' $$f(x - tf'(x)) < f(x) - t\alpha (f'(x))^2$$
#' 
#' Backtracking line search Algorithm
#' ===
#' 
#' - Fix parameter $0 < \beta < 1$ and $0 < \alpha \le 1/2$
#' - At each gradient decent iteration, start with $t=1$, while
#'     $$f(x - t\times f'(x)) > f(x) - \alpha t \times (f'(x))^2$$
#'     Update $t = \beta t$ 
#' - When the above interation stops, use $t$ as the step size
#' 
#' **Note:** In each iteration, you may always want to initialize $t = 1$ before backtracking line search.
#' 
#' 
#' Implementation of back-tracking for gradient decent on the motivating example
#' ===
#' 
#' - Minimize $f(x) = \exp(x) + x^4$
#' - gradient $g(x) = \exp(x) + 4 x^3$
#' - Initial point $x_0 = 1$
#' - initial stepsize $t_0 = 1$, $\alpha = 1/3$, $\beta = 1/2$
#'     - At each step, start with $t = t_0$, 
#'     - if $f(x - tg(x)) > f(x) - \alpha t (g(x))^2$, set $t = \beta t$
#'     - otherwise, use $t$ as step size
#'     
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) exp(x) + x^4 ## original function
g <- function(x) exp(x) + 4 * x^3 ## gradient function
curve(f, from = -1, to = 1, lwd=2) ## visualize the objective function

x <- x0 <- 1; x_old <- 0; t0 <- 1; k <- 0; error <- 1e-6; beta = 0.8; alpha <- 0.4
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error & k < 30){
  k <- k + 1
  x_old <- x
  
  ## backtracking
  t <- t0
  while(f(x_old - t*g(x_old)) > f(x_old) - alpha * t * g(x_old)^2){
    t <- t * beta
  }
  
  x <- x_old - t*g(x_old)
  trace <- c(trace, x) ## collecting results
  points(x, f(x), col=k, pch=19)
  segments(x0=x, y0 = f(x), x1 = x_old, y1 = f(x_old))
}

print(trace)

#' 
#' 
#' 
#' Exercise (will be on HW), solve for logistic regression
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
library(ElemStatLearn)
data2 <- prostate[,c("svi", "lcavol")]
str(data2)

#' 
#' - Y: svi, binary 0/1.
#' - X: lcavol, continuous. 
#' - logistic regression
#' 
#' $$\log \frac{E(Y|x)}{1 - E(Y|x)} = \beta_0 + x\beta_1$$
#' 
#' 
#' 
#' 
#' Exercise (will be on HW), solve for logistic regression
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
glm_binomial_logit <- glm(svi ~ lcavol, data = prostate,  family = binomial())
summary(glm_binomial_logit)

#' 
#' - We set $\beta_0 = -5.0296$.
#' - Try to optimize $\beta_1$, the coefficient for lcavol using gradient decent method
#' - compare with glm result.
#' - compare with the result using optimize() function.
#' 
#' 
#' Exercise (will be on HW), some hints
#' ===
#' 
#' 
#' - density function:
#' $$f(y) = p(x)^y(1 - p(x))^{1 - y}$$
#' 
#' - likelihood function:
#' $$L(\beta_0, \beta_1) = \prod_{i = 1}^n p(x_i)^{y_i}(1 - p(x_i))^{1 - y_i}$$
#' 
#' - logistic regression, logit link
#' $$\log \frac{p(x)}{1 - p(x)} = \beta_0 + x\beta_1$$
#' 
#' 
#' - log likelihood function:
#' 
#' $\begin{aligned}
#' l(\beta_0, \beta_1) &= \log\prod_{i = 1}^n p(x_i)^{y_i}(1 - p(x_i))^{(1 - y_i)} \\ 
#' & = \sum_{i=1}^n y_i \log p(x_i) + (1 - y_i) \log (1 - p(x_i)) \\
#' & = \sum_{i=1}^n \log (1 - p(x_i)) + \sum_{i=1}^n y_i \log \frac{p(x_i)}{1 - p(x_i)} \\
#' & = \sum_{i=1}^n -\log(1 + \exp(\beta_0 + x_i \beta_1)) + \sum_{i=1}^n y_i (\beta_0 + x_i\beta_1)\\
#' \end{aligned}$
#' 
#' 
#' 
#' Newton's method
#' ===
#' 
#' - For unconstrained, smooth univariate convex optimization
#' $$\min f(x)$$
#' where $f$ is convex, twice differentable,
#' $x \in \mathbb{R}$, $f \in \mathbb{R}$.
#' For **gradient descent**, start with intial value $x^{(0)}$ and repeat the following
#' $(k = 1,2,3,\ldots)$ until converge
#' $$x^{(k)} = x^{(k - 1)} - t_k f'(x^{(k - 1)})$$
#' 
#' 
#' - For **Newton's method**,  start with intial value $x^{(0)}$ and repeat the following
#' $(k = 1,2,3,\ldots)$ until converge
#' $$x^{(k)} = x^{(k - 1)} - (f''(x^{(k - 1)}))^{-1} f'(x^{(k - 1)})$$
#' Where $f''(x^{(k - 1)})$ is the second derivative of $f$ at $x^{(k - 1)}$.
#' It is also refered as Hessian matrix for higher dimension (e.g. $x \in \mathbb{R}^p$)
#' 
#' 
#' Newton's method interpretation
#' ===
#' 
#' - For gradient decent step at $x$, we minimize the quadratic approximation
#' $$f(y) \approx f(x) + f'(x)(y - x) + \frac{1}{2t}(y - x)^2$$
#' over $y$, which yield the update $x^{(k)} = x^{(k - 1)} - tf'(x^{(k-1)})$
#' 
#' - Newton's method uses a better quadratic approximation:
#' $$f(y) \approx f(x) + f'(x)(y - x) + \frac{1}{2}f''(x)(y - x)^2$$
#' minimizing over $y$ yield  $x^{(k)} = x^{(k - 1)} - (f''(x^{(k-1)}))^{-1}f'(x^{(k-1)})$
#' 
#' 
#' 
#' Newton's method on our motivating example
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) exp(x) + x^4 ## original function
g <- function(x) exp(x) + 4 * x^3 ## gradient function
h <- function(x) exp(x) + 12 * x^2 ## Hessian function
curve(f, from = -1, to = 1, lwd=2) ## visualize the objective function

x <- x0 <- 0.8; x_old <- 0; t <- 0.1; k <- 0; error <- 1e-6
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error){
  k <- k + 1
  x_old <- x
  x <- x_old - 1/h(x_old)*g(x_old)
  trace <- c(trace, x) ## collecting results
  points(x, f(x), col=k, pch=19)
}
lines(trace, f(trace), lty = 2)

trace_newton <- trace
print(trace)

#' 
#' Compare Gradient decent with Newton's method
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) exp(x) + x^4 ## original function
par(mfrow=c(1,2))
title_grad <- paste("gradient decent, nstep =", length(trace_grad))
curve(f, from = -1, to = 1, lwd=2, main = title_grad) 
points(trace_grad, f(trace_grad), col=seq_along(trace_grad), pch=19)
lines(trace_grad, f(trace_grad), lty = 2)

title_newton <- paste("Newton's method, nstep =", length(trace_newton))
curve(f, from = -1, to = 1, lwd=2, main = title_newton) 
points(trace_newton, f(trace_newton), col=seq_along(trace_newton), pch=19)
lines(trace_newton, f(trace_newton), lty = 2)

#' 
#' Alternative interpretation about Newton's method
#' ===
#' 
#' - Aternative interpretation of Newton's step: we seek a direction $v$ so that $f'(x + v) = 0$.
#' - By Taylor expansion: 
#' $$0 = f'(x + v) \approx f'(x) + f''(x)v$$
#' - Solve for $v$, we have $v = -(f''(x))^{-1} f'(x)$
#' 
#' 
#' ![](../figure/newton2.png)
#' 
#' Backtracking line search for Newton's method
#' ===
#' 
#' - Usually pure Newton's method works very well (do not necessary need to use backtracking)
#' - We have seen pure Newton's method, which need not to decent at all iterations.
#' - In practice, we can use Newton's method with backtracking search, which repeats
#' $$x^{(k)} = x^{(k - 1)} - t(f''(x^{(k-1)}))^{-1}f'(x^{(k-1)})$$
#' - Note that the pure Newton's method uses $t = 1$
#' 
#' 
#' - Algorithm:
#'     - Fix parameter $0 < \beta < 1$ and $0 < \alpha \le 1/2$
#'     - At each Newton's iteration $x$, start with $t=1$, while
#'     $$f(x + t  \Delta x) > f(x) + \alpha t \times f'(x) \Delta x$$
#'     Update $t = \beta t$ 
#'     - In other word (set $\Delta x = - (f''(x))^{-1}  f'(x)$), while
#'     $$f(x - t  (f''(x))^{-1} f'(x)) > f(x) - \alpha t (f''(x))^{-1} (f'(x))^2$$
#'     Update $t = \beta t$
#'     
#' 
#' Implementation of back-tracking for Newton's method
#' ===
#' 
#' - Minimize $f(x) = \exp(x) + x^4$
#' - gradient $g(x) = \exp(x) + 4 x^3$
#' - Hessian $h(x) = \exp(x) + 12 x^2$
#' - Initial point $x_0 = 1$
#' - initial stepsize $t_0 = 1$, $\alpha = 1/3$, $\beta = 1/2$
#'     - At each step, start with $t = t_0$, 
#'     - if $f(x - t  g(x) / h(x) ) > f(x) - \alpha t (g(x))^2/h(x)$, set $t = \beta t$
#'     - otherwise, use $t$ as step size
#'     
## ----------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x) exp(x) + x^4 ## original function
g <- function(x) exp(x) + 4 * x^3 ## gradient function
h <- function(x) exp(x) + 12 * x^2 ## gradient function
curve(f, from = -1, to = 1, lwd=2) ## visualize the objective function

x <- x0 <- 1; x_old <- 0; t0 <- 1; k <- 0; error <- 1e-6; beta = 0.8; alpha <- 0.4
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error & k < 30){
  k <- k + 1
  x_old <- x
  
  ## backtracking
  t <- t0
  while(f(x_old - t*g(x_old)/h(x_old)) > f(x_old) - alpha * t * g(x_old)^2/h(x_old)){
    t <- t * beta
  }
  
  x <- x_old - t*g(x_old)/h(x_old)
  trace <- c(trace, x) ## collecting results
  points(x, f(x), col=k, pch=19)
  segments(x0=x, y0 = f(x), x1 = x_old, y1 = f(x_old))
}

print(trace)

#' 
#' 
#' Exercise (will be on HW), solve for logistic regression
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
library(ElemStatLearn)
glm_binomial_logit <- glm(svi ~ lcavol, data = prostate,  family = binomial())
summary(glm_binomial_logit)

#' 
#' - We set $\beta_0 = -5.0296$.
#' - Try to optimize $\beta_1$, the coefficient for lcavol using Newton's method
#' - compare with gradient method
#' 
#' 
#' 
#' Comparison between gradient decent and Newton's method
#' ===
#' 
#' Method      | Gradient decent  | Newton's method
#' ------------- | ------------- | -------------
#' Order    | First order method | Second Order method
#' Criterion   | smooth $f$      | double smooth $f$
#' Convergence (\# iterations)       | Slow   | Fast
#' Iteration cost  | cheap (compute gradient)  | moderate to expensive (Compute Hessian)
#' 
#' 
#' Multivariate Case
#' ===
#' 
#' - What if $\beta$ is not a scalar but instead a vector (i.e. $\beta \in \mathbb{R}^p$)?
#' 
#' - Will gradient decent still work? 
#'     - Yes
#' 
#' - How to calculate the gradient (derivative) for multivariate case?    
#'     
#' - Will Newton's method still work? 
#'     - Yes
#' 
#' - How to calculate the Hessian matrix for multivariate case?    
#' 
#' 
#' 
#' 
#' Gradient in multivariate case
#' ===
#' 
#' - $\beta = (\beta_1, \beta_2, \ldots, \beta_p)^\top \in \mathbb{R}^p$ is a $p$-dimensional column vector
#' - $f(\beta) \in \mathbb{R}$ is a function which map $\mathbb{R}^p \rightarrow \mathbb{R}$
#' - Suppose $f(\beta)$ is differentiable with respect to $\beta$.
#' 
#' Then 
#' $$\nabla_\beta f(\beta) = \frac{\partial f(\beta)}{\partial \beta} 
#' = (\frac{\partial f(\beta)}{\partial \beta_1}, \frac{\partial f(\beta)}{\partial \beta_2}, \ldots, 
#' \frac{\partial f(\beta)}{\partial \beta_p})^\top \in \mathbb{R}^p$$
#' 
#' - Example
#' $$f(x,y) = 4x^2 + y^2 + 2xy - x - y$$
#'     - $\frac{\partial f}{\partial x} = 8x + 2y - 1$
#'     - $\frac{\partial f}{\partial y} = 2y +2x - 1$
#'     - $\nabla f = (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})^\top = 
#'     \binom{8x + 2y - 1}{2y +2x - 1}  \in \mathbb{R}^2$
#' 
#' Visualization of the example function
#' ===
#' 
## ---- cache=T----------------------------------------------------------------------------------------------------------------------------------
f <- function(x,y){
  4 * x^2 + y^2 + 2 * x * y - x - y 
}


xx <- yy <- seq(-20, 20, len=1000)
zz <- outer(xx, yy, f)
# Contour plots
contour(xx,yy,zz, xlim = c(-10,10), ylim = c(-20,20), nlevels = 10, levels = seq(1,100,length.out = 10), main="Contour Plot")

#' 
#' Hessian in multivariate case
#' ===
#' 
#' - $\beta = (\beta_1, \beta_2, \ldots, \beta_p)^\top \in \mathbb{R}^p$ is a $p$-dimensional column vector
#' - $f(\beta) \in \mathbb{R}$ is a function which map $\mathbb{R}^p \rightarrow \mathbb{R}$
#' - Suppose $f(\beta)$ is twice differentiable with respect to $\beta$.
#' 
#' Then 
#' $$\nabla_\beta f(\beta) = \frac{\partial f(\beta)}{\partial \beta} 
#' = (\frac{\partial f(\beta)}{\partial \beta_1}, \frac{\partial f(\beta)}{\partial \beta_2}, \ldots, 
#' \frac{\partial f(\beta)}{\partial \beta_p})^\top \in \mathbb{R}^p$$
#' 
#' $$\Delta_\beta f(\beta) = \nabla^2_\beta f(\beta) = \nabla_\beta\frac{\partial f(\beta)}{\partial \beta} 
#' = (\nabla_\beta \frac{\partial f(\beta)}{\partial \beta_1}, \nabla_\beta \frac{\partial f(\beta)}{\partial \beta_2}, \ldots, 
#' \nabla_\beta \frac{\partial f(\beta)}{\partial \beta_p})^\top
#' $$
#' 
#' $$\Delta_\beta f(\beta)
#' =
#' \begin{pmatrix}
#' \frac{\partial^2 f(\beta)}{\partial \beta_1^2} & \frac{\partial^2 f(\beta)}{\partial \beta_1 \partial \beta_2} & \ldots & \frac{\partial^2 f(\beta)}{\partial \beta_1 \partial\beta_p}\\ 
#' \frac{\partial^2 f(\beta)}{\partial \beta_2 \partial \beta_1 } & \frac{\partial^2 f(\beta)}{\partial \beta_2^2} & \ldots & \frac{\partial^2 f(\beta)}{\partial \beta_2 \partial \beta_p} \\
#' \ldots &\ldots &\ldots &\ldots\\
#' \frac{\partial^2 f(\beta)}{\partial \beta_p \partial \beta_1} & \frac{\partial^2 f(\beta)}{\partial \beta_p \partial \beta_2} & \ldots & \frac{\partial^2 f(\beta)}{\partial\beta_p^2}
#' \end{pmatrix}
#' $$
#' 
#' Hessian in multivariate case example
#' ===
#' 
#' $$f(x,y) = 4x^2 + y^2 + 2xy - x - y$$
#' 
#' - Gradient
#' 
#'     - $\frac{\partial f}{\partial x} = 8x + 2y - 1$
#'     - $\frac{\partial f}{\partial y} = 2y +2x - 1$
#' 
#' $$\nabla f = (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y})^\top = 
#'     \binom{8x + 2y - 1}{2y +2x - 1}  \in \mathbb{R}^2$$
#' 
#' - Hessian
#'     - $\frac{\partial^2 f}{\partial x^2} = 8$
#'     - $\frac{\partial^2 f}{\partial y^2} = 2$
#'     - $\frac{\partial^2 f}{\partial x \partial y} = 2$
#' 
#' $$\Delta f = \begin{pmatrix}
#' 8 & 2 \\
#' 2 & 2
#' \end{pmatrix}
#' \in \mathbb{R}^{2\times 2}$$
#' 
#' 
#' Gradient decent for multivariate case
#' ===
#' 
#' **Gradient decent**: choose initial $\beta^{(0)} \in \mathbb{R}^p$, repeat:
#' $$\beta^{(k)} = \beta^{(k - 1)} - t\times \nabla_\beta f(\beta^{(k - 1)}), k = 1,2,3,\ldots$$
#' 
#' - Until $\frac{\|\beta^{(k)} - \beta^{(k - 1)}\|_2}{\|\beta^{(k)} + \beta^{(k - 1)}\|_2} < \varepsilon$, (e.g. $\varepsilon = 10^{-6}$).
#' - $\nabla_\beta f(\beta^{(k - 1)})$ is the gradient (derivative of $f$) evaluated at $\beta^{(k - 1)}$.
#' - $t$ is a step size for gradient decent procedure.
#' 
#' 
#' 
#' 
#' Gradient decent on our motivating example
#' ===
#' 
## ---- cache=T----------------------------------------------------------------------------------------------------------------------------------
f <- function(x,y){
  4 * x^2 + y^2 + 2 * x * y - x - y 
}

g <- function(x,y){
  c(8*x + 2*y - 1, 2*y +2*x - 1)
}

xx <- yy <- seq(-20, 20, len=1000)
zz <- outer(xx, yy, f)
# Contour plots
contour(xx,yy,zz, xlim = c(-10,10), ylim = c(-20,20), nlevels = 10, levels = seq(1,100,length.out = 10), main="Contour Plot")


x <- x0 <- 8; x_old <- 0; 
y <- y0 <- -10; y_old <- 0; 
curXY <- c(x,y)
preXY <- c(x_old, y_old)
t <- 0.1; k <- 0; error <- 1e-6
trace <- list(curXY)
points(x0,y0, col=1, pch=1)

l2n <- function(avec){
  sqrt(sum(avec^2))
}
diffChange <- function(avec, bvec){
  deltaVec <- avec - bvec
  sumVec <- avec + bvec
  l2n(deltaVec)/l2n(sumVec)
}

while(diffChange(curXY, preXY) > error){
  k <- k + 1
  preXY <- curXY
  curXY <- preXY - t*g(preXY[1], preXY[2])
  trace <- c(trace, list(curXY)) ## collecting results
  points(curXY[1], curXY[2], col=k, pch=19)
  segments(curXY[1], curXY[2], preXY[1], preXY[2])
}
print(k)

#' 
#' 
#' 
#' How to select step size $t$, backtracking line search
#' ===
#' 
#' ![](../figure/backTracking.png)
#' 
#' 
#' 
#' How to select step size $t$, backtracking line search
#' ===
#' 
#' - Linear approximation (first order Taylor expansion) at the current point $x$.
#' $$f(x + t\Delta x) = f(x) + t\nabla f(x)^\top\Delta x$$
#' 
#' - Decrease the slope of the approximation by $\alpha$
#' 
#' $$f(x) + t\alpha \nabla f(x)^\top\Delta x$$
#' 
#' - A sufficient condition for a proposed $t$ to guarantee **decent** (decrease objective function):
#' $$f(x + t\Delta x) < f(x) + t\alpha \nabla f(x)^\top\Delta x$$
#' 
#' - Plugin $\Delta x = -f'(x)$
#' $$f(x - t\nabla f(x)) < f(x) - \alpha t \times \|(\nabla f(x)\|_2^2$$
#' 
#' - Algorithm:
#'     - Fix parameter $0 < \gamma < 1$ and $0 < \alpha \le 1/2$
#'     - At each gradient decent iteration, start with $t=1$, while
#'     $$f(\beta - t\times \nabla f(\beta)) > f(\beta) - \alpha t \times \|(\nabla f(\beta)\|_2^2$$
#'     Update $t = \gamma t$ 
#' 
#' 
#' Implementation of back-tracking for gradient decent
#' ===
#' 
#' - Minimize $f(x,y) = 4x^2 + y^2 + 2xy - x - y$
#' - gradient $\nabla f = \binom{8x + 2y - 1}{2y +2x - 1}$
#' - Initial point $x_0 = 8$; $y_0 = -10$
#' - initial step size $t_0 = 1$, $\alpha = 1/3$, $\gamma = 1/2$
#'     - At each step, start with $t = t_0$, 
#'     - if $f(x - tg(x)) > f(x) - \alpha t \|g(x)\|_2^2$, set $t = \gamma t$
#'     - otherwise, use $t$ as step size
#'     
## ---- cache=T----------------------------------------------------------------------------------------------------------------------------------
f0 <- function(x,y){
  4 * x^2 + y^2 + 2 * x * y - x - y 
}

f <- function(avec){
  x <- avec[1]
  y <- avec[2]
  4 * x^2 + y^2 + 2 * x * y - x - y 
}

g <- function(avec){
  x <- avec[1]
  y <- avec[2]  
  c(8*x + 2*y - 1, 2*y +2*x - 1)
}

x <- y <- seq(-20, 20, len=1000)
z <- outer(x, y, f0)
# Contour plots
contour(x,y,z, xlim = c(-10,10), ylim = c(-20,20), nlevels = 10, levels = seq(1,100,length.out = 10), main="Contour Plot")


x <- x0 <- 8; x_old <- 0; 
y <- y0 <- -10; y_old <- 0; 
curXY <- c(x,y)
preXY <- c(x_old, y_old)
t0 <- 1; k <- 0; error <- 1e-6
alpha = 1/3
beta = 1/2
trace <- list(curXY)
points(x0,y0, col=1, pch=1)

l2n <- function(avec){
  sqrt(sum(avec^2))
}
diffChange <- function(avec, bvec){
  deltaVec <- avec - bvec
  sumVec <- avec + bvec
  l2n(deltaVec)/l2n(sumVec)
}

while(diffChange(curXY, preXY) > error){
  k <- k + 1
  preXY <- curXY
  
  ## backtracking
  t <- t0
  while(f(preXY - t*g(preXY)) > f(preXY) - alpha * t * l2n(g(preXY))^2){
    t <- t * beta
  }
  
  curXY <- preXY - t*g(preXY)
  trace <- c(trace, list(curXY)) ## collecting results
  points(curXY[1], curXY[2], col=k, pch=19)
  segments(curXY[1], curXY[2], preXY[1], preXY[2])
  
}
print(k)

#' 
#' 
#' Exercise, solve for logistic regression
#' ===
#' 
## ---- cache=T----------------------------------------------------------------------------------------------------------------------------------
library(ElemStatLearn)
glm_binomial_logit <- glm(svi ~ lcavol, data = prostate,  family = binomial())
summary(glm_binomial_logit)

#' 
#' - Optimize $\beta_1$, the coefficient for lcavol and the intercept $\beta_0$ simultaneously using gradient decent method.
#' 
#' 
#' Newton's method for multivariate case
#' ===
#' 
#' - For unconstrained, smooth univariate convex optimization
#' $$\min f(\beta)$$
#' where $f$ is convex, twice differentiable,
#' $\beta \in \mathbb{R}^p$, $f \in \mathbb{R}$.
#' For **gradient descent**, start with initial value $\beta^{(0)}$ and repeat the following
#' $(k = 1,2,3,\ldots)$ until converge
#' $$\beta^{(k)} = \beta^{(k - 1)} - t_k \nabla f(\beta^{(k - 1)})$$
#' 
#' 
#' - For **Newton's method**,  start with initial value $\beta^{(0)}$ and repeat the following
#' $(k = 1,2,3,\ldots)$ until converge
#' $$\beta^{(k)} = \beta^{(k - 1)} - (\Delta f(\beta^{(k - 1)}))^{-1} \nabla f(\beta^{(k - 1)})$$
#' Where $\Delta f(\beta^{(k - 1)}) \in \mathbb{R}^{p\times p}$ is the Hessian matrix of $f$ at $\beta^{(k - 1)}$.
#' 
#' 
#' Implement multivariate Newton's method
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
f0 <- function(x,y){
  4 * x^2 + y^2 + 2 * x * y - x - y 
}

f <- function(avec){
  x <- avec[1]
  y <- avec[2]
  4 * x^2 + y^2 + 2 * x * y - x - y 
}

g <- function(avec){
  x <- avec[1]
  y <- avec[2]  
  c(8*x + 2*y - 1, 2*y +2*x - 1)
}

h <- function(avec){
  x <- avec[1]
  y <- avec[2]  
  res <- matrix(c(8,2,2,2),2,2)  ## Hessian function
  return(res)  
} 

x <- y <- seq(-20, 20, len=1000)
z <- outer(x, y, f0)
# Contour plots
contour(x,y,z, xlim = c(-10,10), ylim = c(-20,20), nlevels = 10, levels = seq(1,100,length.out = 10), main="Contour Plot")


x <- x0 <- 8; x_old <- 0; 
y <- y0 <- -10; y_old <- 0; 
curXY <- c(x,y)
preXY <- c(x_old, y_old)
t0 <- 1; k <- 0; error <- 1e-6
trace <- list(curXY)
points(x0,y0, col=1, pch=1)

l2n <- function(avec){
  sqrt(sum(avec^2))
}
diffChange <- function(avec, bvec){
  deltaVec <- avec - bvec
  sumVec <- avec + bvec
  l2n(deltaVec)/l2n(sumVec)
}

while(diffChange(curXY, preXY) > error){
  k <- k + 1
  preXY <- curXY
  
  curXY <- preXY - solve(h(curXY)) %*% g(preXY)
  trace <- c(trace, list(curXY)) ## collecting results
  points(curXY[1], curXY[2], col=k, pch=19)
  segments(curXY[1], curXY[2], preXY[1], preXY[2])
}

k ## why k is only 2?

#' 
#' Exercise
#' ===
#' 
#' $$f(x,y) = \exp(xy) + y^2 + x^4$$
#' 
#' What are the x and y such that $f(x,y)$ is minimized?
#' 
#' - Use Newton's method to solve this problem
#' 
#' 
#' 
#' Backtracking line search for Newton's method
#' ===
#' 
#' - Usually pure Newton's method works very well (do not necessary need to use backtracking)
#' - We have seen pure Newton's method, which need not to decent at all iterations.
#' - In practice, we can use Newton's method with backtracking search, which repeats
#' $$\beta^{(k)} = \beta^{(k - 1)} - t (\Delta f(\beta^{(k - 1)}))^{-1} \nabla f(\beta^{(k - 1)})$$
#' - Note that the pure Newton's method uses $t = 1$
#' 
#' 
#' - Algorithm:
#'     - Fix parameter $0 < \gamma < 1$ and $0 < \alpha \le 1/2$
#'     - At each Newton's iteration $x$, start with $t=1$, while
#'     $$f(\beta + t  \Delta \beta) > f(\beta) + \alpha t \times \nabla f(\beta)^\top \Delta \beta$$
#'     Update $t = \gamma t$ 
#'     - In other word (set $\Delta \beta = - (\Delta f(\beta))^{-1} \nabla f(\beta)$), while
#'     $$f(\beta - t \Delta f(\beta)^{-1} \nabla f(\beta)) 
#'     > f(\beta) - \alpha t (\nabla f(\beta))^\top (\Delta f(\beta))^{-1} \nabla f(\beta)$$
#'     Update $t = \gamma t$
#'     
#' 
#' 
#' Exercise, solve for logistic regression
#' ===
#' 
## ---- cache=T----------------------------------------------------------------------------------------------------------------------------------
library(ElemStatLearn)
glm_binomial_logit <- glm(svi ~ lcavol, data = prostate,  family = binomial())
summary(glm_binomial_logit)

#' 
#' - Optimize $\beta_1$, the coefficient for lcavol and the intercept $\beta_0$ simultaneously using Newton's method.
#' 
#' 
#' Coordinate decent
#' ===
#' 
#' - Suppose $\beta \in \mathbb{R}^P$ and $f(\beta)$ is a mapping from $\mathbb{R}^P$ to $R$.
#' - We can use **coordinate descent** to find a minimizer.
#' - Start with an initial guess $\beta^{(0)} = (\beta^{(0)}_1, \beta^{(0)}_2, \ldots, \beta^{(0)}_p)^\top$ and repeat the following:
#'     - $\beta_1^{(k)} = \arg \min_{\beta_1} f(\beta_1, \beta_2^{(k-1)}, \beta_3^{(k-1)}, \ldots, \beta_p^{(k-1)})$
#'     - $\beta_2^{(k)} = \arg \min_{\beta_2} f(\beta_1^{(k)}, \beta_2, \beta_3^{(k-1)}, \ldots, \beta_p^{(k-1)})$
#'     - $\beta_3^{(k)} = \arg \min_{\beta_3} f(\beta_1^{(k)}, \beta_2^{(k)}, \beta_3, \ldots, \beta_p^{(k-1)})$
#'     - $\ldots$
#'     - $\beta_p^{(k)} = \arg \min_{\beta_p} f(\beta_1^{(k)}, \beta_2^{(k)}, \beta_3^{(k)}, \ldots, \beta_p)$
#'     
#' - Continue with  $k=1,2,3,\ldots$ until converge.
#'     - Note that after $\beta_j^{(k)}$ has been updated, we use the new value for future update.
#'     
#'     
#' Example on linear regression
#' ===
#' 
#' - Consider linear regression problem:
#' 
#' $$ \min_{\beta \in \mathbb{R}^p} f(\beta) = \min_{\beta \in \mathbb{R}^p} \frac{1}{2} \|y - X\beta\|_2^2$$
#' where $y \in \mathbb{R}^n$ and $X \in \mathbb{R}^{n \times p}$ with columns $X_1$ (intercept), $X_2$, $\ldots$, $X_p$.
#' 
#' - Minimizing over $\beta_j$ while fixing all $\beta_i$, $i \ne j$:
#' $$0 = \frac{\partial f(\beta)}{\partial \beta_j} = X_j^\top (X\beta - y) =  X_j^\top (X_j\beta_j + X_{-j}\beta_{-j} - y)$$
#' 
#' - We can solve for the coordinate descent updating rule:
#' 
#' $$\beta_j = \frac{X_j^\top (y - X_{-j}\beta_{-j})}{X_j^\top X_j}$$
#' 
#' implement coordinate decent using prostate cancer data
#' ===
#' 
## ---- cache=T----------------------------------------------------------------------------------------------------------------------------------
library(ElemStatLearn)

y <- prostate$lcavol
x0 <- prostate[,-match(c("lcavol", "train"), colnames(prostate))]
x <- cbind(1, as.matrix(x0))

beta <- rep(0,ncol(x))
beta_old <- rnorm(ncol(x))

error <- 1e-6
k <- 0
while(diffChange(beta, beta_old) > error){
  k <- k + 1
  beta_old <- beta
  for(j in 1:length(beta)){
    xj <- x[,j]
    xj_else <- x[,-j]
    beta_j_else <- as.matrix(beta[-j])
    beta[j] <- t(xj) %*% (y - xj_else %*% beta_j_else) / sum(xj^2)
  }
}

print(beta)

## compare with lm
lm(lcavol ~ . - train, data=prostate)

#' 
#' 
#' 
#' Reference
#' ===
#' 
#' <http://www.stat.cmu.edu/~ryantibs/convexopt/>
#' 
