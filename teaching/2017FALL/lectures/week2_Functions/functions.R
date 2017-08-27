#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "R function"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday August 28, 2017"
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
sum
body(sum)
formals(sum)
environment(sum)

#' 
#' Example - Odds Ratio Function
#' ===
#' - Suppose we want to write a function that calculates the odds ratio $\theta$ of a 2 x 2 contingency table and the asymptotic confidence interval for $\theta$.
#' 
#' ![](https://caleb-huo.github.io/teaching/2017FALL/lectures/week2_Functions/figure/oddsRatioTable.png)
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
## ------------------------------------------------------------------------
X <- matrix(c(189, 104, 10845, 10933), nrow=2,
            dimnames=list(Treatment=c("Placebo","Aspirin"), 
                          "Myocardial Infarction"=c("Yes", "No")))
X

#' 
#' Returning Objects
#' ===
#' - Often we will want a function to return an object that can be assigned. Two functions for returning objects are return() and invisible(). 
#' - The return() function prints and returns its arguments. 
#' - If the end of a function is reached without calling return(), the value of the last evaluated expression is returned and outputted.
#' - You can also omit {} if there is only one line for the body
#' 
## ------------------------------------------------------------------------
odds.ratio0 <- function(X){
  result <- X[1,1]*X[2,2]/(X[1,2]*X[2,1])
  return(result)
}
odds.ratio0(X)

#' 
#' Returning Objects
#' ===
#' - If the end of a function is reached without calling return(), the value of the last evaluated expression is returned and outputted.
#' - You can also omit {} if there is only one line for the body
#' 
## ------------------------------------------------------------------------
odds.ratio1 <- function(X){
  X[1,1]*X[2,2]/(X[1,2]*X[2,1])
}
odds.ratio1(X)
odds.ratio2 <- function(X) ## if the function has only one line, don't even need brackets
  X[1,1]*X[2,2]/(X[1,2]*X[2,1])

odds.ratio2(X)

#' 
#' 
#' Returning Objects 2
#' ===
#' - The invisible() function is useful when we want a function to return values which can be assigned, but which do not print when they are not assigned.
## ------------------------------------------------------------------------
odds.ratio3 <- function(X){
  result <- X[1,1]*X[2,2]/(X[1,2]*X[2,1])
  invisible(result)
}
odds.ratio3(X)
## here nothing is returned
OR <- odds.ratio3(X)
OR

#' 
#' 
#' Return multiple variables
#' ===
#' - A list is often a good tool for returning multiple objects.
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
x <- 15
f <- function() x
f()

x <- 20
f()

#' Every operation is a function call 
#' ===
#' 
#' - infix operators: +, -, *, /
## ------------------------------------------------------------------------
x <- 10; y <- 5; 
x + y
'+'(x, y)

#' 
#' - Control flow operators: for, if while
## ------------------------------------------------------------------------
for(i in 1:2) cat(i,' ')
'for'(i, 1:2, cat(i,' '))

#' 
#' - Subsetting operator
## ------------------------------------------------------------------------
x <- 1:10; x[3]
'['(x,3)

#' 
#' 
#' Other infix functions
#' ===
#' 
## ------------------------------------------------------------------------
## %%: Remainder operator
5 %% 3
## %/%: Integer division
5 %/% 3
## %*% Matrix multiplication
## to be discussed next week.

## %o% Outer product
1:2 %o% 1:3

## %in% Match operator
c(1,2,3,4,5,6) %in% c(2,6,8)


#' 
#' User defined infix functions
#' ===
#' 
## ------------------------------------------------------------------------
`%divisible%` <- function(x,y)
{
   if (x%%y ==0) return (TRUE)
   else          return (FALSE)
}
24 %divisible% 5
24 %divisible% 4

#' 
## ------------------------------------------------------------------------
`%mypaste%` <- function(x,y)
{
   paste0(x, y)
}
"a" %mypaste% "b"

#' 
#' Function arguments -- default value
#' ===
#' - default argument of a function can be defined in the function formals
## ------------------------------------------------------------------------
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
#'     - partial name.
#'     - by position
#' - Arguments match order 
#'     1. first by exact name (perfect matching), 
## ------------------------------------------------------------------------
f <- function(arg, brg=10) 
  list(arg = arg, brg = brg)
str(f(arg=2, brg=10))

#' 
#' ---
#' 
#'     2. then by prefix matching, 
## ------------------------------------------------------------------------
f <- function(arg, brg=10) 
  list(arg = arg, brg = brg)
str(f(b=10, ar=2))

#'     3. finally by position matching
## ------------------------------------------------------------------------
f <- function(arg, brg=10) 
  list(a = arg, b = brg)
str(f(2, 10))

#' 
#' 
#' function calling habits
#' ===
#' - good calls
## ------------------------------------------------------------------------
mean(1:10); mean(1:10, trim = 0.05)

#' 
#' - overkill
## ------------------------------------------------------------------------
mean(x=1:10)

#' 
#' - confusing
## ------------------------------------------------------------------------
mean(1:10, n = T); mean(1:10, , FALSE); mean(1:10, 0.05); mean(, TRUE, x = c(1:10, NA))

#' 
#' 
#' Extra argument: ...
#' ===
#' 
#' - ...: allows variability in the arguments given to R functions
#' - First check the function argument of plot()
#' - We will create a new function which will pass extra argument to another function
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
do.call(mean, list(1:10, na.rm=TRUE))
mean(1:10, na.rm=TRUE)

#' 
#' 
#' R path function
#' ===
#' 
#' - getwd(): get current working directory
## ------------------------------------------------------------------------
getwd()

#' 
#' - setwd(): set current working directly
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
dir()
dir("~")

#' 
#' ---
#' 
#' - file.exist(): check a certain filename already
#' 
## ------------------------------------------------------------------------
file.exists("result.Rdata")
file.exists("amatrix.csv")

#' 
#' - dir.create(): create a new folder
## ------------------------------------------------------------------------
dir.create("tmp")
dir.create("tmp") # there is a warning message if the folder already exist.

#' 
#' 
#' File operation -- Write
#' ===
#' - write.csv: by default, csv is comma delimited
## ------------------------------------------------------------------------
amatrix <- matrix(1:6,3,2)
rownames(amatrix) <- paste0("row", 1:nrow(amatrix))
colnames(amatrix) <- paste0("col", 1:ncol(amatrix))
write.csv(amatrix, file = "amatrix.csv")

#' 
#' 
#' - write.table, txt is tab delimited
## ------------------------------------------------------------------------
write.table(amatrix, file = "amatrix.txt")
write.table(amatrix, file = "amatrix2.txt", row.names = FALSE) ## suppress row names
write.table(amatrix, file = "amatrix3.txt", quote = FALSE) ## suppress quote for strings
write.table(amatrix, file = "amatrix4.txt", row.names = FALSE, quote = FALSE) ## suppress both

#' 
#' 
#' - save
## ------------------------------------------------------------------------
amatrix <- matrix(1:6,3,2)
save(amatrix, file='amatrix.rdata')

#' - writeLines
## ------------------------------------------------------------------------
aline <- "I like Biostatistical computing"
zz <- file("writeLines.txt", "w") ## w for wright
writeLines(aline, con=zz)
close(zz)

#' 
#' 
#' read
#' ===
#' - read.csv: 
## ------------------------------------------------------------------------
matrix_csv <- read.csv("amatrix.csv")
matrix_csv

matrix_csv2 <- read.csv("amatrix.csv", row.names = 1)
matrix_csv2

#' 
#' ---
#' 
#' - read.table, txt is tab delimited
## ------------------------------------------------------------------------
matrix_table <- read.table("amatrix4.txt")
matrix_table
matrix_table2 <- read.table("amatrix4.txt", header=TRUE)
matrix_table2

#' 
#' ---
#' 
#' - load
## ------------------------------------------------------------------------
load("amatrix.rdata")
amatrix
matrix_load <- get(load("amatrix.rdata"))
matrix_load

#' 
#' ---
#' 
#' - readLines
## ------------------------------------------------------------------------
dir()
zz <- file("writeLines.txt", "r") ## r for read
readLines(con=zz)
close(zz)

#' 
#' reading a text file in R line by line
#' ===
## ------------------------------------------------------------------------
inputFile <- "amatrix4.txt"
con  <- file(inputFile, open = "r")

while (length(oneLine <- readLines(con, n = 1, warn = FALSE)) > 0) {
    print(oneLine)
  } 

close(con)

#' 
#' 
#' Time function - Keep track the time of your progress
#' ===
#' - Sys.time()
#' 
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
f <- function(x, y=x){
  x <- x + 1
  return(y)
}
f(1)

#' - Since there is no argument for y, R evaluates the default value of y only when it is needed. 
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
function(x=4) g(x) + h(x)
formals(function(x=4) g(x) + h(x))
body(function(x=4) g(x) + h(x))
environment(function(x=4) g(x) + h(x))

#' 
#' Call functions from another library
#' ===
#' - You can use functions from other packages by library
## ------------------------------------------------------------------------
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
## ------------------------------------------------------------------------
e <- new.env()
e$a <- FALSE
e$b <- "a"
e$c <- 2:3
e$d <- 1:3

#' ![](https://caleb-huo.github.io/teaching/2017FALL/lectures/week2_Functions/figure/environment.png)
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
## ------------------------------------------------------------------------
search() ## lists all parents of the global environment.
parent.env(globalenv())

#' 
#' Load a new package
#' ===
## ------------------------------------------------------------------------
library(mclust)
search() ## lists all parents of the global environment.

#' 
#' Load a new package 2
#' ===
#' 
## ------------------------------------------------------------------------
environment(combn)
library(combinat)
search() ## lists all parents of the global environment.

#' 
#' ---
#' 
## ------------------------------------------------------------------------
environment(combn)
environment(utils::combn)
environment(combinat::combn)

#' 
#' - Use :: to specify namespace
#' 
#' An experiment
#' ===
## ------------------------------------------------------------------------
x <- matrix(1:4,nrow=2,ncol=2)
nrow
nrow(x)
dim(x)
dim <- function() c(1,1)
## guess what is nrow(x)

#' 
#' ---
#' 
## ------------------------------------------------------------------------
nrow(x) 
environment(nrow)
environment(dim)
environment(base::nrow)

#' 
#' 
#' Functional programming
#' ===
#' 
#' R is a functional programming (FP) language. You can do anything with functions that you can do with vectors: 
#' 
#' - assign them to variables
#' - store them in lists
#' - pass them as arguments to other functions
#' - create them inside functions
#' - return them as the result of a function
#' 
#' Closures
#' ===
#' 
#' - Closures are functions written by functions
#' - Closures get their name because they enclose the environment of the parent function and can access all its variables.
#' - Two levels of parameters: a parent level controls operation and a child level that does the work.
## ------------------------------------------------------------------------
power <- function(exponent){
  function(x)
    x^exponent
} ## parent level
## child level
square <- power(2)
square(2); 
cube <- power(3)
cube(2); 

#' 
#' 
#' 
#' Top-Down function design
#' ===
#' - Break the original problem into smaller sub-problems.
#' - Sub-problems are in turn broken down into even smaller sub-problems.
#' - continuing until all sub-problems are solved.
#' 
#' 
#' Code sketch
#' ===
#' Below is pseudocode for top-down function design
## ------------------------------------------------------------------------
original.problem <- function(many.args){
  sub1.problem.result <- step.sub1(some.args)
  sub2.problem.result <- step.sub2(some.args, sub1.problem.result)
  final.result <- step.sub3(some.args, sub2.problem.result)
  return(final.result)
}

step.sub1 <- function(some.args){
  ## blabla
}

step.sub2 <- function(some.args, sub1.problem.result){
  ## blabla
}

step.sub3 <- function(some.args, sub2.problem.result){
  ## blabla
}


#' 
#' 
#' 
#' References
#' ===
#' 
#' - [http://adv-r.had.co.nz](http://adv-r.had.co.nz)
#' - [http://www.stat.cmu.edu/~ryantibs/statcomp/](http://www.stat.cmu.edu/~ryantibs/statcomp/)
#' 
#' - reference code generation
## ---- echo=TRUE, eval=TRUE-----------------------------------------------
suppressWarnings(library(knitr))
purl("functions.rmd", output = "functions.R", documentation = 2)

#' 
#' 
#' In class exercises
#' ===
#' 
#' 1. Constructive confidence interval for one sided t test, verifying arguments.
#' 
#' 2. Read in a file, print the some element of each line.
#' 
#' 3. Read functional programming example <http://adv-r.had.co.nz>.
#' 
#' Confidence interval for one sample t.test
#' ===
#' 
#' Suppose you have a vector of numbers $X$ from Gaussian distribution.
#' Please write a function to construct $\alpha$ level confidence interval for the mean estimate of $X$, mimicing the ratio.ratio function.
#' [one sample t.test reference](http://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)
#' 
#' - Input arguments include a vector of numbers $X$ and $\alpha$ level with default value 0.05.
#' - Verify that the argument X is non-missing, X should contain at least 5 element and X should be numeric type.
#' - Keep track of the progress by print() or cat()
#'     - evaluating mean estimate and its standard error
#'     - constructing CI at level $\alpha$
#'     - outputing result
#' - return the mean estimate, standard error and $\alpha$ level confidence interval.
#' - test your function with the following simulated data, keep track of how long time does it take to evaluate this function
#' 
## ------------------------------------------------------------------------
n <- 20
x <- rnorm(n, 0, 1) ## generate n=20 samples from N(0,1)

## yourFunction(x, alpha = 0.05)

#' 
#' Read in data
#' ===
#' 
#' - The iris data is available online.
#'     - https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
#'     
#' - Read in this data
#'     - save ith element of each line in a new vector, where i equals to the remainder of line number divided by 4. (E.g. i = 1 for line 1, i = 2 for line 2, i = 3 for line 3, i = 4 for line 5, i = 1 for line 5...). 
#'     - The total number of elements in the new vector equals to the total number of lines.
#' 
## ------------------------------------------------------------------------
newVec <- NULL

#' 
#' 
#' Functional programming
#' ===
#' 
#' Why functional programming is useful
#' 
#' - Read <http://adv-r.had.co.nz/Functional-programming.html> 
#'     - about Case study: numerical integration
#' 
