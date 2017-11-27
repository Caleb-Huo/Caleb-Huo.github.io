#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday Nov 27, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: multi-variate optimization
#' ---
#' 
#' 
#' Outline
#' ===
#' 
#' - Recap of Univariate Optimization
#' - Multi-variate Optimization
#'     - Gradient decent
#'     - Newton's method
#'     - coordinate decent
#' - Optimization problem with constraint
#' - Multi-variate Optimization using R package
#' 
#' Univariate Gradient decent
#' ===
#' - Consider an unconstrained minimization of $f$, we want to find $\beta^*$ such that 
#' $$f(\beta^*) = \min_{\beta\in \mathbb{R}} f(\beta)$$
#' - Another way to formulate the problem is to find $\beta^*$ such that 
#' $$\beta^* = \arg_{\beta\in \mathbb{R}} \min f(\beta)$$
#' 
#' **Gradient decent**: choose initial $\beta^{(0)} \in \mathbb{R}$, repeat:
#' $$\beta^{(k)} = \beta^{(k - 1)} - t\times f'(\beta^{(k - 1)}), k = 1,2,3,\ldots$$
#' 
#' - Until $|\beta^{(k)} - \beta^{(k - 1)}| < \varepsilon$, (e.g. $\varepsilon = 10^{-6}$).
#' - $f'(\beta^{(k - 1)})$ is the gradient (derivative of $f$) evaluated at $\beta^{(k - 1)}$.
#' - $t$ is a step size for gradient decent procedure.
#' 
#' 
#' Interpretation
#' ===
#' - At each iteration $k$, consider Taylor expansion at $\beta^{(k - 1)}$:
#' $$f(y) = f(\beta^{(k - 1)}) + f'(\beta^{(k - 1)}) \times (y - \beta^{(k - 1)}) + \frac{1}{2} f''(\beta^{(k - 1)}) (y - \beta^{(k - 1)})^2 + \ldots$$
#' - Quadratic approximation:
#'     - Replacing $f''(\beta^{(k - 1)})$ with $\frac{1}{t}$
#'     - Ignore higher order terms
#' 
#' $$f(y) \approx f(\beta^{(k - 1)}) + f'(\beta^{(k - 1)}) \times (y - \beta^{(k - 1)}) + \frac{1}{2t}  (y - \beta^{(k - 1)})^2$$
#' 
#' - Minimizing the quadratic approximation.
#' $$\beta^{(k)} = \arg_y \min f(y)$$
#'     
#'     - Set $f'(\beta^{(k)}) = 0 \Leftrightarrow f'(\beta^{(k - 1)}) + \frac{1}{t}  (\beta^{(k)} - \beta^{(k - 1)}) = 0 \Leftrightarrow \beta^{(k)} = \beta^{(k - 1)} - t\times f'(\beta^{(k - 1)})$
#' 
#'     - This is exactly the same as the gradient decent procedure.
#' 
#' 
#' Visual interpretation
#' ===
#' 
#' ![](../figure/GDprocedure.png)
#' 
#' 
#' - iterate 1: $\beta^{(1)} = \beta^{(0)} - t\times f'(\beta^{(0)})$
#'     - quadratic approximation at <span style="color:red">$\beta_0$</span>, 
#'     - minimizer of the quadratic approximation is <span style="color:blue">$\beta_1$</span>
#' 
#' - iterate 2: $\beta^{(2)} = \beta^{(1)} - t\times f'(\beta^{(1)})$
#'     - quadratic approximation at <span style="color:blue">$\beta_1$</span>, 
#'     - minimizer of the quadratic approximation is <span style="color:green">$\beta_2$</span>
#' 
#' - ...
#' 
#' 
#' 
#' Newton's method
#' ===
#' 
#' - For unconstrained, smooth univariate convex optimization
#' $$\min f(\beta)$$
#' where $f$ is convex, twice differentiable,
#' $\beta \in \mathbb{R}$, $f \in \mathbb{R}$.
#' For **gradient descent**, start with initial value $\beta^{(0)}$ and repeat the following
#' $(k = 1,2,3,\ldots)$ until converge
#' $$\beta^{(k)} = \beta^{(k - 1)} - t_k f'(\beta^{(k - 1)})$$
#' 
#' 
#' - For **Newton's method**,  start with initial value $\beta^{(0)}$ and repeat the following
#' $(k = 1,2,3,\ldots)$ until converge
#' $$\beta^{(k)} = \beta^{(k - 1)} - (f''(\beta^{(k - 1)}))^{-1} f'(\beta^{(k - 1)})$$
#' Where $f''(\beta^{(k - 1)})$ is the second derivative of $f$ at $\beta^{(k - 1)}$.
#' It is also referred as Hessian matrix for higher dimension (e.g. $\beta \in \mathbb{R}^p$)
#' 
#' 
#' Newton's method interpretation
#' ===
#' 
#' - For gradient decent step at $\beta$, we minimize the quadratic approximation
#' $$f(y) \approx f(\beta) + f'(\beta)(y - \beta) + \frac{1}{2t}(y - \beta)^2$$
#' over $y$, which yield the update $\beta^{(k)} = \beta^{(k - 1)} - tf'(\beta^{(k-1)})$
#' 
#' - Newton's method uses a better quadratic approximation:
#' $$f(y) \approx f(\beta) + f'(\beta)(y - \beta) + \frac{1}{2}f''(\beta)(y - \beta)^2$$
#' minimizing over $y$ yield  $\beta^{(k)} = \beta^{(k - 1)} - (f''(\beta^{(k-1)}))^{-1}f'(\beta^{(k-1)})$
#' 
#' 
#' New problem?
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
## ---- cache=T------------------------------------------------------------
f <- function(x,y){
  4 * x^2 + y^2 + 2 * x * y - x - y 
}


x <- y <- seq(-20, 20, len=1000)
z <- outer(x, y, f)
# Contour plots
contour(x,y,z, xlim = c(-10,10), ylim = c(-20,20), nlevels = 10, levels = seq(1,100,length.out = 10), main="Contour Plot")

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
## ---- cache=T------------------------------------------------------------
f <- function(x,y){
  4 * x^2 + y^2 + 2 * x * y - x - y 
}

g <- function(x,y){
  c(8*x + 2*y - 1, 2*y +2*x - 1)
}

x <- y <- seq(-20, 20, len=1000)
z <- outer(x, y, f)
# Contour plots
contour(x,y,z, xlim = c(-10,10), ylim = c(-20,20), nlevels = 10, levels = seq(1,100,length.out = 10), main="Contour Plot")


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
#' Divergence of Gradient decent
#' ===
#' 
## ---- cache=T------------------------------------------------------------
f <- function(x,y){
  4 * x^2 + y^2 + 2 * x * y - x - y 
}

g <- function(x,y){
  c(8*x + 2*y - 1, 2*y +2*x - 1)
}

B <- 1000
plot(x,xlim=c(-B,B), ylim=c(-B,B))
#contour(x,y,z, xlim = c(-100,100), ylim = c(-200,200), nlevels = 10, levels = seq(1,10000,length.out = 10), main="Contour Plot")


x <- x0 <- 0.8; x_old <- 0; 
y <- y0 <- -0.10; y_old <- 0; 
curXY <- c(x,y)
preXY <- c(x_old, y_old)
t <- 0.8; k <- 0; error <- 1e-6
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

## repeat the following chunk of code to see divergence of gradient decent
k <- k + 1
preXY <- curXY
curXY <- preXY - t*g(preXY[1], preXY[2])
points(curXY[1], curXY[2], col=k, pch=19)
segments(curXY[1], curXY[2], preXY[1], preXY[2])

#' 
#' 
#' How to select step size $t$, backtracking line search
#' ===
#' 
#' ![](../figure/backTracking.png)
#' 
#' 
#' How to select step size $t$, backtracking line search
#' ===
#' 
#' - Purpose, we want to select $t$ such that $f(\beta - t\times \nabla f(\beta)) < f(\beta)$
#'     - $\beta$ is current position.
#'     - $t$ is step size.
#' - A sufficient condition:
#'     - $f(\beta - t\times \nabla f(\beta)) < f(\beta) - \alpha t \times \|(\nabla f(\beta)\|_2^2$
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
## ---- cache=T------------------------------------------------------------
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
## ---- cache=T------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
#' Divergence in Newton's method
#' ===
#' 
#' - Here's an example of Newton's method for root finding, generated from the polynomial $z^4 - 1$, so that the four roots of the function are at 1, -1, i and -i.
#' 
#' - ![Whole picture](https://www.chiark.greenend.org.uk/~sgtatham/newton/simple.png)
#'   ![Zoomed picture](https://www.chiark.greenend.org.uk/~sgtatham/newton/zoomed.png)
#' 
#' Note: same color denotes the region where will lead to the same root.
#' Reference: <https://www.chiark.greenend.org.uk/~sgtatham/newton/>
#' 
#' - Here's another example, generated from the polynomial $(z^2 + 1)(z^2 - 2.3^2)$,
#' - ![](https://www.chiark.greenend.org.uk/~sgtatham/newton/holes.png)
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
## ---- cache=T------------------------------------------------------------
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
## ---- cache=T------------------------------------------------------------
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
#' One dimensional lasso problem
#' ===
#' 
#' - $y \in \mathbb{R}$
#' - $\beta \in \mathbb{R}$
#' 
#' $$\arg \min_\beta f(\beta) = \arg \min_\beta \frac{1}{2}(y - \beta)^2 + \lambda |\beta|$$
#' 
#' - gradient set to 0:
#' 
#' $$0 = \frac{\partial f(\beta)}{\partial \beta} = \beta - y + \lambda \frac{\partial |\beta|}{\partial  \beta}$$
#' 
#' - Solution by conditions:
#'     - When $\beta > 0$, $0 = \beta - y + \lambda$, $\beta = y - \lambda$, we have $y > \lambda$
#'     - When $\beta < 0$, $0 = \beta - y - \lambda$, $\beta = y + \lambda$, we have $y < - \lambda$
#'     - Also, when $-\lambda \le y \le \lambda$, $\beta = 0$
#' 
#' 
#' - Therefore, one dimensional lasso solution is
#' 
#' $$\beta = S_\lambda(y) = \begin{cases}
#' y - \lambda \text{ if } y > \lambda\\
#' y + \lambda \text{ if } y < - \lambda\\
#' 0 \text{ if } -\lambda \le y \le \lambda\\
#' \end{cases}$$
#' 
#' soft thresholding operator and hard thresholding operator
#' ===
#' 
#' - $S_\lambda(y)$ is called soft thresholding operator 
#' $$S_\lambda(y) = \begin{cases}
#' y - \lambda \text{ if } y > \lambda\\
#' y + \lambda \text{ if } y < - \lambda\\
#' 0 \text{ if } -\lambda \le y \le \lambda\\
#' \end{cases}$$
#' 
#' - Hard thresholding operator is defined as
#' $$H_\lambda(y) = \begin{cases}
#' y  \text{ if } y > \lambda\\
#' y  \text{ if } y < - \lambda\\
#' 0 \text{ if } -\lambda \le y \le \lambda\\
#' \end{cases}$$
#' 
#' ![](https://www.researchgate.net/profile/Thai_V_Hoang/publication/258161038/figure/fig3/AS:392761867030541@1470653134190/Fig-6-Illustrations-of-the-hard-thresholding-and-soft-thresholding-operators-O-is-a.png)
#' 
#' 
#' Exercise
#' ===
#' 
#' - Using coordinate decent to solve the lasso problem with $p$ variables.
#' - You may need the soft thresholding operator to derive updating rule.
#' - Apply to prostate cancer data using svi as outcome variable; compare the result with lars package.
#' 
#' 
#' Constrained optimization - Lagrange multiplier
#' ===
#' 
#' - Consider ridge regression problem
#'     - $y \in \mathbb{R}^n$
#'     - $\beta \in \mathbb{R}^p$
#'     - $X \in \mathbb{R}^{n \times p}$
#'     - $f(\beta) =  \|y - X\beta\|_2^2$
#'     - $\arg \min_x \|y - X\beta\|_2^2$ such that $\|\beta\|_2^2 = t$
#'     
#' - Write the Lagrange function using Lagrange multiplier method:
#' $$L(\beta) =  \|y - X\beta\|_2^2 + \lambda (\|\beta\|_2^2 - t) $$
#' - Set $\frac{\partial L(\beta)}{\partial \theta} = 0$
#' $$X^\top(X\beta - y) + \lambda\beta = 0$$
#' $$\beta(\lambda) = (X^\top X + \lambda I)^{-1} X^\top y$$
#' - solution:
#'     - $\|\beta(\lambda)\|^2_2 = t$, solve for $\lambda$ (via binary search), then solve for $\beta$
#' 
#' 
#' Constrained optimization - Karush-Kuhn-Tucher conditions
#' ===
#' 
#' Given general problem
#' 
#' - $\min f(x)$ subject to 
#'     - $h_i(x) \le 0$, $i = 1, 2, \ldots, m$
#'     - $l_j(x) = 0$, $j = 1, 2, \ldots, r$
#' 
#' - Lagrange function:
#' 
#' $$L(x) = f(x) + \sum_{i=1}^m u_i h_i(x) + \sum_{j=1}^r v_j l_j(x) $$
#' 
#' The **Karush-Kuhn-Tucher conditions** or **KKT conditions** are:
#' 
#' - stationary: 
#' $$0 = \frac{\partial f(x)}{\partial x} + \sum_{i=1}^m u_i \frac{\partial h_i(x)}{\partial x} + \sum_{j=1}^r v_j \frac{\partial l_j(x)}{\partial x}$$
#' - complementary slackness: 
#' $$u_i h_i(x) =0, \forall i$$
#' - primal feasibility:
#' $$h_i(x) \le 0, l_j(x) = 0, \forall i,j$$
#' - dual feasibility:
#' $$u_i \ge 0, \forall i$$
#' 
#' Application: water-filling (B & V page 245)
#' ===
#' 
#' - Consider the following problem
#'     - $\min_{x\in R^n} - \sum_{i=1}^n \log(\alpha_i + x_i)$ 
#'     - subject to $x \ge 0$, $1^\top x = 1$ 
#' 
#' - write down Lagrange function
#'     $$L(x) = - \sum_{i=1}^n \log(\alpha_i + x_i) - u_i x_i + v(1^\top x - 1) $$
#' - Applying KKT condition and get
#'     - $-\frac{1}{\alpha_i + x_i} - u_i + v = 0$, for $i = 1, \ldots, n$
#'     - $u_i x_i = 0$, for $i = 1, \ldots, n$
#'     - $x \ge 0$
#'     - $1^\top x = 1$
#'     - $u_i \ge 0$, for $i = 1, \ldots, n$
#' 
#' 
#' 
#' Application: water-filling (B & V page 245)
#' ===
#' 
#' - Eliminate $u$:
#'     - $\frac{1}{\alpha_i + x_i} \le v$, $i = 1, \ldots, n$
#'     - $x_i (v - \frac{1}{\alpha_i + x_i}) = 0$, $i = 1, \ldots, n$
#'     
#' - Therefore either
#'     - $x_i = \frac{1}{v} - \alpha_i$, if $v < \frac{1}{\alpha_i}$
#'     - $x_i = 0$, if $v \ge \frac{1}{\alpha_i}$
#' 
#' - $x = \max \{0, 1/v - \alpha_i \}$
#' 
#' - We need $x$ to be feasible, $1^\top x = 1$, so
#' $$\sum_{i=1}^n \max\{0, 1/v - \alpha_i\} = 1$$
#' Which is univariate equation. Can solve $v$ using binary search.
#' 
#' 
#' Optimize this function in R
#' ===
#' 
#' - use function constrOptim
#' - Fix $n = 10$
#' - generate $\alpha_i$ from UNIF distribution
#' - Inequality constraint: $x_i \ge 0$
#' - Equality constraint: $\sum_i x_i = 1$
#'     - $\sum_i x_i - 1 \ge 0$
#'     - $\sum_i x_i - 1 \le 0$
#' 
#' 
#' - Constraints:
#'     - $I x - 0 \ge 0$
#'     - $1^\top x - 1 \ge 0$    
#'     - $- 1^\top x + 1 \ge 0$    
#' 
#' Optimize this function in R
#' ===
#' 
## ------------------------------------------------------------------------
n <- 10
alpha <- runif(n)
x0 <- runif(n)
x1 <- x0/sum(x0)
f <- function(x){
  -sum(log(alpha + x))
}
grad <- function(x){
  -1/(alpha + x)
}

ui1 <- diag(n)
ui2 <- rep(1,n)
ui3 <- rep(-1,n)
ui <- rbind(ui1, ui2, ui3)

ci1 <- rep(0,n)
ci2 <- 1 - 1e-12
ci3 <- -1 - 1e-12
ci <- c(ci1,ci2,ci3)


constrOptim(x1, f, grad, ui, ci)

#' 
#' 
#' Exercise
#' ===
#' 
#' - Solve the water filling problem using KKT condition
#' - Compare the result with the result from constrOptim function
#' 
#' Reference
#' ===
#' 
#' - convex optimization <http://www.stat.cmu.edu/~ryantibs/convexopt-S15/>
#' - B & V <https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf>
#' 
## ------------------------------------------------------------------------
knitr::purl("multiOptimization.rmd", output = "multiOptimization.R ", documentation = 2)

