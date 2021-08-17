#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "R function"
#' author: Zhiguang Huo (Caleb)
#' date: "Wed Sep 2nd, 2020"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' Why do we need functions
#' ===
#' - If you repeat a procedure more than 2 times, you should consider make it as a function.
#'     - Save your work in the future.
#'     - Avoid mistakes.
#'     
#'     
#' 
#' Component of a R function
#' ===
#' 
#' - All R functions have three parts:
#'     - body(), the code inside the function
#'     - formals(), list of arguments which controls how you can call the function.
#'     - environment(), the map of the location of the function's variables.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
cubic <- function(x){
  return(x^3)
} 
## x: function argument
## cubic: function name

body(cubic)
formals(cubic)
environment(cubic)

#' 
#' 
#' 
#' Primitive functions
#' ===
#' - Primitive funcitons are exceptions to the rule.
#' - Primitive funcitons, like sum(), call C code directly .Primitive() and contain no R code.
#' - Primitive funcitons are only in the base package, they can be more efficient.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
sum
body(sum)
formals(sum)
environment(sum)

#' 
#' Example - Odds Ratio Function
#' ===
#' - Suppose we want to write a function that calculates the odds ratio $\theta$ of a 2 x 2 contingency table and the asymptotic confidence interval for $\theta$.
#' 
#' ![](../figure/oddsRatioTable.png)
#' 
#' 
#' - Sample odds ratio, 
#'    $$\hat{\theta}=\frac{ad}{bc} $$ 
#' - The asymptotic standard error for $log(\hat{\theta})$ is $$SE(log\hat{\theta})=\sqrt{\frac{1}{a}+\frac{1}{b}+\frac{1}{c}+\frac{1}{d}} $$ 
#' - The asymptotic 100(1-$\alpha$)\% confidence interval for $log\hat{\theta}$ is, $$ log\hat{\theta} \pm z_{\alpha/2}SE(log\hat{\theta}) $$
#' - Exponentiate the upper and lower bounds to get a confidence interval for $\theta$.
#'    
#'     
#' 
#' Example - Odds Ratio Function
#' ===
#' - Consider the following data that describes the relationship between myocardial infarction and aspirin use (Agresti 1996). 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
X <- matrix(c(189, 104, 10845, 10933), nrow=2,
            dimnames=list(Treatment=c("Placebo","Aspirin"), 
                          "Myocardial Infarction"=c("Yes", "No")))
X

#' 
#' Returning Objects
#' ===
#' - Often we will want a function to return an object that can be assigned. The function for returning objects is return(). 
#' - The return() function prints and returns its arguments. 
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
odds.ratio0 <- function(X){
  result <- X[1,1]*X[2,2]/(X[1,2]*X[2,1])
  return(result)
}
odds.ratio0(X)

#' 
#' Returning Objects
#' ===
#' - If the end of a function is reached without calling return(), the value of the last evaluated expression is returned and outputted.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
odds.ratio1 <- function(X){
  X[1,1]*X[2,2]/(X[1,2]*X[2,1])
}
odds.ratio1(X)

#' 
#' - You can also omit {} if there is only one line for the body
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
odds.ratio2 <- function(X) ## if the function has only one line, don't even need brackets
  X[1,1]*X[2,2]/(X[1,2]*X[2,1])

odds.ratio2(X)

#' 
#' 
#' 
#' Return multiple variables
#' ===
#' - A list is often a good tool for returning multiple objects.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
odds.ratio <- function(X, conf.level=0.95){
  OR <- X[1,1]*X[2,2]/(X[1,2]*X[2,1])
  logOR.SE <- sqrt(sum(1/X))
  
  alpha <- 1 - conf.level
  CI.lower <- exp(log(OR) - qnorm(1 - alpha/2)*logOR.SE)
  CI.upper <- exp(log(OR) + qnorm(1 - alpha/2)*logOR.SE)
  # qnorm returns the quantiles of a Gaussian distribution.
  
  out <- list(OR=OR, CI=c(CI.lower, CI.upper), conf.level=conf.level)
  return(out)
}
odds.ratio(X)
odds.ratio(X)$OR

#' 
#' tracking progress of a function
#' ===
#' - The cat() and print() functions can be used to output results; 
#'     - the cat() function gives more control over the appearance of the output. 
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
odds.ratio <- function(X, conf.level=0.95){
  cat("calculating odds ratio...","\n")
  OR <- X[1,1]*X[2,2]/(X[1,2]*X[2,1])
  logOR.SE <- sqrt(sum(1/X))
  
  cat("calculating confidence interval...","\n")
  alpha <- 1 - conf.level
  CI.lower <- exp(log(OR) - qnorm(1 - alpha/2)*logOR.SE)
  CI.upper <- exp(log(OR) + qnorm(1 - alpha/2)*logOR.SE)
  
  cat("done, returning output...","\n")
  out <- list(OR=OR, CI=c(CI.lower, CI.upper), conf.level=conf.level)
  return(out)
}
out <- odds.ratio(X)

#' 
#' 
#' Verifying Arguments
#' ===
#' - If you are writing a function for others to use it is often a good idea to include code that verifies that the appropriate arguments were entered.
#' - If an argument value is not valid we want to stop executing the expression and return an error message.
#'      - missing(): Can be used to test whether a value was specified as an argument to a function; returns TRUE if a value is not specified and FALSE if a value is specified.
#'      - stop(): Stop execution of the current expression and prints an error.
#'      - warning(): Generate a warning message.
#'      - message(): Generate a diagnostic message.
#'      - stopifnot(): If any of the arguments are not all TRUE then stop() is called and an error message is produced that indicates the first element of the argument list that is not TRUE. 
#' 
#' examples for verifying arguments
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------------
odds.ratio <- function(X, conf.level=0.95){
  stopifnot(!missing(X), is.matrix(X),dim(X)==c(2,2),X>0)

  cat("calculating odds ratio...","\n")
  OR <- X[1,1]*X[2,2]/(X[1,2]*X[2,1])
  logOR.SE <- sqrt(sum(1/X))
  
  cat("calculating confidence interval...","\n")
  alpha <- 1 - conf.level
  CI.lower <- exp(log(OR) - qnorm(1 - alpha/2)*logOR.SE)
  CI.upper <- exp(log(OR) + qnorm(1 - alpha/2)*logOR.SE)
  
  cat("done, returning output...","\n")
  out <- list(OR=OR, CI=c(CI.lower, CI.upper), conf.level=conf.level)
  return(out)
}
out <- odds.ratio(X)

#' 
#' 
#' Try catch
#' ===
#' 
#' tryCatch function allows your code to be excuted even error exists
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
# result = tryCatch({
#     expr
# }, warning = function(w) {
#     warning-handler-code
# }, error = function(e) {
#     error-handler-code
# }
# )

#' 
#' - success
## ----------------------------------------------------------------------------------------------------------------------------------------------------
result = tryCatch({
  2^6
}, warning = function(w) {
  print(paste("MY_WARNING:  ",w))
  return("warning")
}, error = function(e) {
  print(paste("MY_ERROR:  ",e))
  return("error")
}
)
result

#' 
#' ---
#' 
#' - warning
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
result = tryCatch({
  1:3 + 1:2
}, warning = function(w) {
  print(paste("MY_WARNING:  ",w))
  return("warning")
}, error = function(e) {
  print(paste("MY_ERROR:  ",e))
  return("error")
}
)
result

#' 
#' ---
#' 
#' - error
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
result = tryCatch({
  "A" + "B"
}, warning = function(w) {
  print(paste("MY_WARNING:  ",w))
  return("warning")
}, error = function(e) {
  print(paste("MY_ERROR:  ",e))
  return("error")
}
)
result

#' 
#' 
#' 
#' 
#' Lexical scoping for R functions
#' ===
#' 
#' Scoping is the set of rules that govern how R looks up the value of a symbol.
#' 
#' - Name masking
#' - fresh start
#' - dynamic lookup
#' 
#' Name masking
#' ===
#' 
#' - If a function doesn't find a variable/function inside a function, it will search one level up
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- 1
h <- function(){
  y <- 2
  i <- function(){
    z <- 3
    c(x, y, z)
  }
  i()
}

h()

#' 
#' 
#' A fresh start
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
j <- function(){
  if(!exists("a")){
    a <- 1
  } else {
    a <- a + 1
  }
  print(a)
}
j()
j()

#' 
#' Dynamic lookup
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- 15
f <- function() x
f()

x <- 20
f()

#' Every operation is a function call 
#' ===
#' 
#' - infix operators: +, -, *, /
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- 10; y <- 5; 
x + y
'+'(x, y)

#' 
#' - Control flow operators: for, if while
## ----------------------------------------------------------------------------------------------------------------------------------------------------
for(i in 1:2) cat(i,' ')
'for'(i, 1:2, cat(i,' '))

#' 
#' - Subsetting operator
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- 1:10; x[3]
'['(x,3)

#' 
#' 
#' Other infix functions
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
## %%: Remainder operator
5 %% 3
## %/%: Integer division
5 %/% 3
## %*% Matrix multiplication
## to be discussed in the future.

## %o% Outer product
1:2 %o% 1:3

## %in% Match operator
c(1,2,3,4,5,6) %in% c(2,6,8)


#' 
#' 
#' Function arguments -- default value
#' ===
#' - default argument of a function can be defined in the function formals
## ----------------------------------------------------------------------------------------------------------------------------------------------------
f <- function(arg, brg=10) 
  list(arg = arg, brg = brg)
str(f(arg=2, brg=10))
str(f(arg=2)) 

#' 
#' - revisit = and <-
#'     - = can be used for assigning value and specifying function argument
#'     - <- can only be used for assigning value.
#' 
#' 
#' Function arguments
#' ===
#' - How to specify function arguments
#'     - complete name
#'     - partial name
#'     - by position
#' - Arguments match order 
#'     1. first by exact name (perfect matching), 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
f <- function(arg, brg=10) 
  list(arg = arg, brg = brg)
str(f(arg=2, brg=10))

#' 
#' ---
#' 
#'     2. then by prefix matching, 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
f <- function(arg, brg=10) 
  list(arg = arg, brg = brg)
str(f(b=10, ar=2))

#'     3. finally by position matching
## ----------------------------------------------------------------------------------------------------------------------------------------------------
f <- function(arg, brg=10) 
  list(a = arg, b = brg)
str(f(2, 10))

#' 
#' 
#' function calling habits
#' ===
#' - good calls
## ----------------------------------------------------------------------------------------------------------------------------------------------------
mean(1:10); mean(1:10, trim = 0.05)

#' 
#' - overkill
## ----------------------------------------------------------------------------------------------------------------------------------------------------
mean(x=1:10)

#' 
#' - confusing
## ----------------------------------------------------------------------------------------------------------------------------------------------------
mean(1:10, n = T); mean(1:10, , FALSE); mean(1:10, 0.05); mean(, TRUE, x = c(1:10, NA))

#' 
#' 
#' Extra argument: ...
#' ===
#' 
#' - ...: allows variability in the arguments given to R functions
#' - First check the function argument of plot()
#' - We will create a new function which will pass extra argument to another function
## ----------------------------------------------------------------------------------------------------------------------------------------------------
myPlot <- function(x,y,...){
  plot(x,y,...)
}
myPlot(1:4,1:4,col=1:4)

#' 
#' 
#' calling a function given a list of arguments
#' ===
#' - If you have a list of function arguments: 
#'     - args <- list(1:10, na.rm=TRUE)
#' - How could you then send that list to mean()? 
#'     - Need do.call()
## ----------------------------------------------------------------------------------------------------------------------------------------------------
do.call(mean, list(1:10, na.rm=TRUE))
mean(1:10, na.rm=TRUE)

#' 
#' 
#' R path function
#' ===
#' 
#' - getwd(): get current working directory
## ----------------------------------------------------------------------------------------------------------------------------------------------------
getwd()

#' 
#' - setwd(): set current working directly
## ----------------------------------------------------------------------------------------------------------------------------------------------------
WD0 <- getwd()
setwd("~") ## ~ represent user home directly 
getwd()
setwd(WD0)
getwd()

#' 
#' Directory information
#' ===
#' 
#' - dir(): show all files available under current working directory.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
dir()
dir("~")

#' 
#' ---
#' 
#' - file.exist(): check a certain filename already
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
file.exists("result.Rdata")
file.exists("amatrix.csv")

#' 
#' - dir.create(): create a new folder
## ----------------------------------------------------------------------------------------------------------------------------------------------------
dir.create("tmp")
dir.create("tmp") # there is a warning message if the folder already exist.

#' 
#' - dir.exists(): chech if a directory already exists.
#' 
#' 
#' 
#' Time function - Keep track the time of your progress
#' ===
#' - Sys.time()
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
start <- Sys.time()
sum(1:1e4)
end <- Sys.time()
end - start

start <- Sys.time()
result <- 0
for(i in 1:1e4){
  result <- result + i
}
result
end <- Sys.time()
end - start

#' 
#' Lazy Evaluation
#' ===
#' - R performs **lazy evaluation** of function arguments. The arguments are not evaluated until they are required.
#'     - Lazy evaluation can save time and memory if the arguments are not needed.
#'     - The function force() forces the evaluation of a function argument.
#' - Consider the function 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x, y=x){
  x <- x + 1
  return(y)
}
f(1)

#' - Since there is no argument for y, R evaluates the default value of y only when it is needed. 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
f <- function(x, y=x){
  force(y) # force to evaluate y.
  x <- x + 1
  return(y)
}
f(1)

#' 
#' 
#' Anonymous functions
#' ===
#' 
#' - You use an anonymous function when it's not worth the effort to give it a name
#' - Like all functions in R, anonymous functions have formals(), body() and a parent environment()
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
function(x=4) g(x) + h(x)
formals(function(x=4) g(x) + h(x))
body(function(x=4) g(x) + h(x))
environment(function(x=4) g(x) + h(x))

#' 
#' Call functions from another library
#' ===
#' - You can use functions from other packages by library
## ----------------------------------------------------------------------------------------------------------------------------------------------------
library(survival)

## seek for help about the survival package
help(package=survival)

## get help for a specific function
?coxph

## head(coxph)
## args(coxph)
## coxph


detach("package:survival", unload=TRUE) ## remove survival package from R loaded environment

#' 
#' 
#' Environments
#' ===
#' - Only introduce the basic structure of R environments.
#' - Refer to Chapter 8 of Advanced R book for details.
#' [http://adv-r.had.co.nz/Environments.html](http://adv-r.had.co.nz/Environments.html)
#' 
#' 
#' Basic Environments
#' ===
#' - The job of an environment is to associate, or bind, a set of names to a set of values.
#' - You can think of an environment as a bag of names.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
e <- new.env()
e$a <- FALSE
e$b <- "a"
e$c <- 2:3
e$d <- 1:3

#' ![](../figure/environment.png)
#' 
#' 
#' Basic environments
#' ===
#' 
#' Generally, an environment is similar to a list, with following important exceptions.
#' 
#' - Every object in an environment has a unique name.
#' - The objects in an environment are not ordered.
#' - An environment has a parent.
#' 
#' There are four special environments:
#' 
#' - The globalenv(), or global environemnt, is the interactive workspace which you normally work on. The parent of the global environment is the last package you attached.
#' - The baseenv(), or base environment, is the environment of the base package. Its parent is the empty environment.
#' - emptyenv(), or empty environment, is the ultimate ancestor of all environments, and the only environment without a parent.
#' - environment() is the current environment.
#' 
#' 
#' Environment search path
#' ===
#' 
#' ![](../figure/searchPath.png)
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
search() ## lists all parents of the global environment.
parent.env(globalenv())

#' 
#' Load a new package
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------------
library(mclust)
search() ## lists all parents of the global environment.

#' 
#' Load a new package 2
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
environment(combn)
library(combinat)
search() ## lists all parents of the global environment.

#' 
#' ---
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
environment(combn)
environment(utils::combn)
environment(combinat::combn)

#' 
#' - Use :: to specify namespace
#' 
#' Function utility searching (optional)
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- matrix(1:4,nrow=2,ncol=2)
nrow
nrow(x)
dim(x)
dim <- function() c(1,1)
## guess what is nrow(x)

#' 
#' ---
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
nrow(x) ## A function in a package will always use the utilities within its namespace first
## this is how R prevents external/global functions from affecting utilities functions inside a package
environment(nrow)
environment(dim)
environment(base::dim)

#' 
#' 
#' ![](../figure/dim_ncol.png)
#' 
#' 
#' 
#' 
#' References
#' ===
#' 
#' - [http://adv-r.had.co.nz](http://adv-r.had.co.nz)
#' - [http://www.stat.cmu.edu/~ryantibs/statcomp/](http://www.stat.cmu.edu/~ryantibs/statcomp/)
#' 
#' 
#' 
#' Other exercises and readings
#' ===
#' 
#' 1. Constructive confidence interval for one sided t test, verifying arguments.
#' 
#' 2. Read functional programming example <http://adv-r.had.co.nz>.
