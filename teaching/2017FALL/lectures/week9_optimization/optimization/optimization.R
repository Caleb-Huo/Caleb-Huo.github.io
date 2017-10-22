#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday October 23, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Univariate optimization
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
#' - Root finding
#'     - Root finding in R
#'     - Newton's method
#'     - Binary search
#' 
#' optimization
#' ===
#' - In mathematics, "optimization" or "mathematical programming" refers to the selection of a best element (with regard to some criterion) from some set of available alternatives.
#' 
#' - A typical optimization problem usually consists of maximizing or minimizing a real function (objective function) by systematically choosing input values under certain constraint.
#' 
#' - "Convex programming" studies the case when the objective function is convex (minimization) or concave (maximization) and the constraint set is convex.
#' 
#' Convex set
#' ===
#' 
#' - $S$ is convex if and only if 
#'     - $\forall x,y \in S$
#'     - $tx + (1 - t)y \in S$, where $t \in [0, 1]$
#'     
#' ![](../figure/convexShape.png)
#' 
#' 
#' Convex function
#' ===
#' 
#' Convex function properties:
#' 
#' - Second derivative always greater than 0
#' - Tangent line always underestimate the function
#' - $\forall t \in (0,1)$, $tf(a) + (1 - t)f(b) > f(ta + (1 - t)b)$
#' 
#' 
#' ![](../figure/convex2.png)
#' 
#' 
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
#' Focus on one dimensional (univariate) optimization this week.
#' We will talk about multivariate optimization at the end of this semester.
#' 
#' Univariate optimization example
#' ===
#' Suppose our objective function is 
#' $$f(x) = e^x + x^4$$
#' What is the minimum value of $f(x)$, what is the correponding $x$?
#' 
## ------------------------------------------------------------------------
f <- function(x) exp(x) + x^4
curve(f, from = -1, to = 1)

#' 
#' 
#' Use R optimize() function to perform the optimization
#' ===
## ------------------------------------------------------------------------
f <- function(x) exp(x) + x^4
xmin <- optimize(f, interval = c(-10, 10))
xmin
curve(f, from = -1, to = 1)
points(xmin$minimum, xmin$objective, col="red", pch=19)

#' 
#' 
#' 
#' Do the optimization yourself
#' ===
#' - You don't have the R optimization package at this moment, (unlikely to happen).
#' - You want to customize the optimization procedure such that it is most efficient specific to your problem.
#' - You want to publish a paper which requires you to propose an optimization procedure to solve your proposed problem.
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
## ------------------------------------------------------------------------
f <- function(x) exp(x) + x^4 ## original function
g <- function(x) exp(x) + 4 * x^3 ## gradient function
curve(f, from = -1, to = 1, lwd=2) ## visualize the objective function

x <- x0 <- 0.8; x_old <- 0; t <- 0.1; k <- 0; error <- 1e-6
trace <- x
points(x0, f(x0), col=1, pch=1)

while(abs(x - x_old) > error){
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
#' - If the objective function has multiple local minimum.
#' - Gradient will still work, but will fall into a local minimum (might be global minimum).
#' 
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
## ------------------------------------------------------------------------
f <- function(x) x^2 ## original function
g <- function(x) 2*x ## gradient function
curve(f, from = -10, to = 10, lwd=2) ## visualize the objective function

x <- x0 <- 1; x_old <- 0; t <- 1.1; k <- 0; error <- 1e-6
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
#' Change to another stepsize
#' ===
#' 
#' - Minimize $f(x) = x^2$
#' - Initial point $x_0 = 1$
#' - stepsize $t = 0.2$
#' 
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
#' - Purpose, we want to select $t$ such that $f(x - t\times f'(x)) < f(x)$
#'     - $x$ is current position.
#'     - $t$ is step size.
#' - A sufficient condition:
#'     - $f(x - t\times f'(x)) < f(x) - \alpha t \times (f'(x))^2$
#' - Algorithm:
#'     - Fix parameter $0 < \beta < 1$ and $0 < \alpha \le 1/2$
#'     - At each gradient decent iteration, start with $t=1$, while
#'     $$f(x - t\times f'(x)) > f(x) - \alpha t \times (f'(x))^2$$
#'     Update $t = \beta t$ 
#' 
#' 
#' Implementation of back-tracking for gradient decent
#' ===
#' 
#' - Minimize $f(x) = x^2$
#' - gradient $g(x) = 2x$
#' - Initial point $x_0 = 1$
#' - initial stepsize $t_0 = 1$, $\alpha = 1/3$, $\beta = 1/2$
#'     - At each step, start with $t = t_0$, 
#'     - if $f(x - tg(x)) > f(x) - \alpha t (g(x))^2$, set $t = \beta t$
#'     - otherwise, use $t$ as step size
#'     
## ------------------------------------------------------------------------
f <- function(x) x^2 ## original function
g <- function(x) 2*x ## gradient function
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
## ------------------------------------------------------------------------
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
#' Exercise (will be on HW), solve for logistic regression
#' ===
#' 
## ------------------------------------------------------------------------
library(ElemStatLearn)
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
#' - logistic regression, logit link
#' $$\log \frac{p(x)}{1 - p(x)} = \beta_0 + x\beta_1$$
#' 
#' - density function:
#' $$f(y) = p(x)^y(1 - p)^{1 - y}$$
#' 
#' - likelihood function:
#' $$L(\beta_0, \beta_1) = \prod_{i = 1}^n p(x_i)^{y_i}(1 - p(x_i))^{1 - y_i}$$
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
#' Root finding
#' ===
#' 
#' - A root-finding algorithm is an algorithm for finding roots of continuous functions. 
#' - A root of a function f, from the real numbers to real numbers is a number $x$ such that $f(x) = 0$. 
#' - As, generally, the roots of a function cannot be computed exactly, nor expressed in closed form, root-finding algorithms provide approximations to roots.
#' 
#' 
#' Root finding in R
#' ===
#' 
#' - visualize the function
#' 
## ------------------------------------------------------------------------
afunction <- function(x) cos(x) - x
curve(afunction, from = -10, to=10)

#' 
#' ---
#' 
#' - apply root finding
## ------------------------------------------------------------------------
aroot <- uniroot(function(x) cos(x) - x, lower = -10, upper = 10, tol = 1e-9)$root
curve(afunction, from = -10, to=10)
points(aroot, afunction(aroot), col=2, pch=19)
print(aroot)

#'     
#' Root finding by Newton's method (optional)
#' ===
#' 
#' - $y = f(x) = f(x^{(n)}) + f'(x^{(n)}) (x - x^{(n)})$ set to 0
#' - $f(x^{(n)}) + f'(x^{(n)}) (x^{(n+1)} - x^{(n)})$
#' - $x^{(n+1)} = x^{(n)} - \frac{f(x^{(n)})}{f'(x^{(n)})}$
#' 
#' Objective function and gradient
#' 
#' - $f(x) = \cos(x) - x$
#' - $g(x) = -\sin(x) - 1$
#' 
## ------------------------------------------------------------------------
tol <- 1e-9
f <- function(x) cos(x) - x
g <- function(x) -sin(x) - 1

niter <- 1; maxiter <- 100
x0 <- 4
x <- x0
x_old <- rnorm(1)
while(abs(x - x_old) > tol & niter < maxiter){
  x_old <- x
  x <- x_old - f(x_old)/g(x_old)
  niter <- niter + 1
}

curve(f, from = -10, to=10)
points(x, f(x), col=2, pch=19)
print(x)

#' 
#' 
#' Root finding by Binary search
#' ===
#' 
## ------------------------------------------------------------------------
curve(afunction, from = -10, to=10)
L <- -8; R <- 8
points(L, afunction(L), col=3, pch=19)
points(R, afunction(R), col=4, pch=19)

#' 
#' - We see that $f(L) > 0$, $f(R) < 0$
#' 
#' - Repeat the following procedure
#'     - If $f(\frac{L+R}{2}) > 0$, update $L = \frac{L+R}{2}$
#'     - If $f(\frac{L+R}{2}) < 0$, update $R = \frac{L+R}{2}$
#'     - stop until $L - R < \varepsilon$
#' - return $\frac{L+R}{2}$
#' 
#' ---
#' 
## ------------------------------------------------------------------------
tol <- 1e-9
f <- function(x) cos(x) - x
L <- -8; R <- 9
niter <- 1; maxiter <- 100

while(abs(L - R) > tol & niter < maxiter){
  fmid <- f((L+R)/2)
  if(fmid>0){
    L <- (L+R)/2
  } else {
    R <- (L+R)/2
  }
  niter <- niter + 1
}

root_binary <- (L+R)/2
curve(f, from = -10, to=10)
points(root_binary,f(root_binary),col=2,pch=19)
print(root_binary)

#' 
#' 
#' Reference
#' ===
#' 
#' <http://www.stat.cmu.edu/~ryantibs/convexopt/>
#' 
## ------------------------------------------------------------------------
knitr::purl("optimization.Rmd", output = "optimization.R ", documentation = 2)

#' 
