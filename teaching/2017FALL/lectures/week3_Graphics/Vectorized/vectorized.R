#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday September 6, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Vectorized calculation
#' ---
#' 
#' Why do we need vectorized calculation (motivating example 1)
#' ===
#' - Elegant and efficient
#' - R is slow in loops but fast for vectorized calculation.
## ------------------------------------------------------------------------
a <- 1:1000000
### version A, loop
start <- Sys.time()
meanA <- 0
for(i in seq_along(a)){
  meanA <- meanA + a[i]/length(a)
}
end <- Sys.time() 
end - start
meanA

### version B, vectorized calculation
start <- Sys.time()
mean(a)
end <- Sys.time() 
end - start


#' 
#' 
#' Why do we need vectorized calculation (motivating example 2)
#' ===
#' - Elegant and efficient
#' - R is slow in loops but fast for vectorized calculation.
## ------------------------------------------------------------------------
a <- 1:1000000
b <- 1000000:1
### version A, loop
start <- Sys.time()
result <- numeric(length(a)) ## create a vector with length length(a) and all elements 0
for(i in seq_along(a)){
  result[i] <- a[i] + b[i]
}
end <- Sys.time() 
end - start

### version B, vectorized calculation
start <- Sys.time()
result <- a + b
end <- Sys.time() 
end - start

#' 
#' Simple examples of vectorized calculation
#' ===
#' - vector algebra
## ------------------------------------------------------------------------
a <- c(1.1,3.4,9.5)
b <- c(9,2,0.8)
a + b ## vector addition
a * b ## vector multiplication
a ^ b ## 1.1^9, 3.4^2, 9.5^0.8


#'   
#' 
#' Simple examples of vectorized calculation 2
#' ===
#' - vector with scalor
## ------------------------------------------------------------------------
a <- 1:8
a + 2
a + c(0,1)
a + c(1,2,3) ## warming message 

#' 
#' 
#' Other scientific calculation
#' ===
## ------------------------------------------------------------------------
a <- seq(1,3,1)
sin(a)
tan(a)
log(a,base = 10)
log10(a)
exp(a)

#' 
#' 
#' vectorized input for plot
#' ===
#' - all vector input
## ------------------------------------------------------------------------
plot(x = 1:10, y = 1:10, col=1:10)

#' 
#' - vector input + scalor input
## ------------------------------------------------------------------------
plot(x = 1:10, y = 1:10, col=1)

#' 
#' Initialize a vector
#' ===
## ------------------------------------------------------------------------
n <- 5
a_int <- integer(n)
a_int

b_int <- rep(0, n)
b_int


c_int <- replicate(5,0)
c_int


#' 
#' 
#' Initialize a vector, other types
#' ===
## ------------------------------------------------------------------------
a_double <- double(n)
a_double


## ------------------------------------------------------------------------
a_char <- character(n)
a_char


## ------------------------------------------------------------------------
a_logical <- logical(n)
a_logical

#' 
#' 
#' A sequence
#' ===
## ------------------------------------------------------------------------
a <- seq(from=10, to=100, by=2)
a
b <- seq(from=10, to=100, length=10)
b
c <- seq(from=10, to=100, along.with=1:10)
c

#' 
#' 
#' 
#' Random number generator
#' ===
#' - Random numbers from normal distribution
## ------------------------------------------------------------------------
set.seed(32611) ## set a seed number such that the random numbers will keep the same
rnorm(n = 5, mean = 0, sd = 1)

#' - mean and sd parameter can also be vectorized.
## ------------------------------------------------------------------------
set.seed(32611)
rnorm(n = 5, mean = 0:4, sd = 1)

#' - Random numbers from 1:10
## ------------------------------------------------------------------------
set.seed(32611)
sample(1:10,1)

#' - Random numbers from Uniform distribution U(0,1)
## ------------------------------------------------------------------------
set.seed(32611)
runif(n = 4)

#' 
#' replicate
#' ===
#' - replicate empty list
## ------------------------------------------------------------------------
replicate(4,list())

#' - a complicate example by replicate
## ------------------------------------------------------------------------
replicate(5, runif(sample(1:5,1)))

#' 
#' 
#' Matrix calculation
#' ===
## ------------------------------------------------------------------------
a <- matrix(1:6,nrow=3,ncol=2)
a ## matrix will be filled by column by default. 
b <- matrix(6:1, nrow=3, ncol=2)
a * b ## similar to vector, matrix algebra will be done element-wise.
a + c(1,2,3) ## if a matrix add a vector, add by column

#' 
#' 
#' 
#' 
#' 
#' apply function: works on margins of a matrix (1)
#' ===
## ------------------------------------------------------------------------
a <- matrix(1:6,nrow=3,ncol=2)
a
apply(a,1,sum) ## for each row, equivalent to rowSums
apply(a,1,var) ## for each row, calculate the variance
apply(a,1,function(x) x^2) ## can also use an anonymous function, note that R will always fill the result by column.

#' 
#' 
#' apply function: works on margins of a matrix (2)
#' ===
## ------------------------------------------------------------------------
apply(a,2,sum) ## for each column, equivalent to colSums
colSums(a)
apply(a,2,mean) ## for each column, equivalent to colMeans
apply(a,2,function(x) min(x)) ## equivalent to apply(a,2,min)
apply(a,2,min)

#' 
#' 
#' Sweep
#' ===
#' 
#' - sweep() allows you to “sweep” out the values of a summary statistic. It is often used with apply() to standardise arrays. 
## ------------------------------------------------------------------------
set.seed(32611)
(x <- matrix(rnorm(6,0,4),nrow=2))
x1 <- sweep(x,1,apply(x,1,min),'-')
x1
x2 <- sweep(x1,1,apply(x1,1,max),'/')
x2

#' 
#' 
#' lapply()
#' ===
#' - lapply will apply a function to each element of a list
#' - lapply() is written in C for performance, 
#' 
## ------------------------------------------------------------------------
l <- list(c(1:10), list(a='a'), c("dog","cat","gator"))
unlist(lapply(l, length))

#' 
#' - a simple R implementation that does the same thing:
#' 
## ------------------------------------------------------------------------
lapply2 <- function(x, f, ...){
  out <- vector("list", length(x))
  for(i in seq_along(x)){
    out[[i]] <- f(x[[i]], ...)
  }
  out
}
unlist(lapply2(l, length))

#' 
#' 
#' lapply usages
#' ===
## ------------------------------------------------------------------------
alist <- list(a = 25, b = 100, c = 64)

#' - loop over the elements: for (x in xs)
## ------------------------------------------------------------------------
lapply(alist, function(x) sqrt(x))

#' - loop over the numeric indices: for (i in seq_along(xs))
## ------------------------------------------------------------------------
lapply(seq_along(alist), function(x) sqrt(alist[[x]]))

#' - loop over the names: for (nm in names(xs))
## ------------------------------------------------------------------------
lapply(names(alist), function(x) sqrt(alist[[x]]))

#' 
#' 
#' lapply on data.frame
#' ===
#' - data.frame shares the property of a list and a matrix
## ------------------------------------------------------------------------
aframe <- data.frame(col1=1:3,col2=4:6)
lapply(aframe, mean) ## as a list
apply(aframe, 2, mean) ## as a matrix


#' 
#' 
#' lapply on a list of functions
#' ===
## ------------------------------------------------------------------------
compute_mean <- list(
  base = function(x) mean(x),
  sum = function(x){
    sum(x)/length(x)},
  mannual = function(x){
    total <- 0
    n <- length(x)
    for(i in seq_along(x)){
      total <- total + x[i]/n
    }
    total
  }
)

set.seed(32611)
x <- runif(1e5)
lapply(compute_mean,function(f) system.time(f(x)))


#' 
#' 
#' sapply() and vapply()
#' ===
#' - sapply() is similar to lapply except they simplify the output to produce an atomic vector.
## ------------------------------------------------------------------------
aframe <- data.frame(col1=1:3,col2=4:6)
sapply(aframe, sum) ## if not the same atomic type, will coerce to a list

#' - vapply() is simimar to sapply but takes an additional argument specifying the output type.
## ------------------------------------------------------------------------
aframe <- data.frame(col1=1:3,col2=4:6)
vapply(aframe, sum, numeric(1)) 

#' 
#' 
#' 
#' 
#' 
#' aggregate
#' ===
#' 
## ------------------------------------------------------------------------
head(state.x77)
head(state.region)
aggregate(state.x77, list(Region = state.region), mean)

#' 
#' tapply
#' ===
#' 
#' - tapply() is a generalisation to apply by groups
## ------------------------------------------------------------------------
pulse <- round(rnorm(22,70,10/3) + rep(c(0,5), c(10,12)))
groups <- rep(c("A", "B"), c(10, 12))
tapply(pulse, groups, length)
tapply(pulse, groups, mean)
tapply2 <- function(x, group, f, ..., simplify = TRUE){
  pieces <- split(x, group)
  sapply(pieces, f, simplify=simplify)
}
tapply2(pulse, groups, length)

#' 
#' Multiple inputs (Map)
#' ===
#' - Map(): handle when more than only one arguments of the function varies.
#' - question: how to calculated weight mean?
## ------------------------------------------------------------------------
xs <- replicate(3, runif(4),simplify=FALSE) ## simplify = TRUE (default) will convert a list to matrix whenever possible
ws <- replicate(3, rnorm(4, 1) + 1,simplify=FALSE)
xs

ws

#' 
#' ---
#' 
#' - approach 1
## ------------------------------------------------------------------------
unlist(lapply(seq_along(xs), function(i){
  weighted.mean(xs[[i]], ws[[i]])
}))

#' 
#' - approach 2
## ------------------------------------------------------------------------
Map(weighted.mean,xs,ws)

#' 
#' - extra argument
## ------------------------------------------------------------------------
unlist(Map(function(x,w) weighted.mean(x, w, na.rm=TRUE),xs, ws))

#' 
#' 
#' approach 3
#' ===
#' 
## ------------------------------------------------------------------------
mapply(weighted.mean,xs,ws)

mapply(function(x,y) weighted.mean(x,y), xs,ws)

#' 
#' 
#' Common Higher-Order Functions in Functional Programming Languages
#' ===
#' 
#' - Reduce uses a binary function to successively combine the elements of a given vector and a possibly given initial value. 
#' - Filter extracts the elements of a vector for which a predicate (logical) function gives true. 
#' - Find and Position give the first or last such element and its position in the vector, respectively. - Map applies a function to the corresponding elements of given vectors. 
#' 
#' 
#' 
#' Reduce
#' ===
#' 
#' - Reduce() reduces a vector, x, to a single value by recursively calling a function, f, two arguments at a tile.
#' - Reduce(f, 1:3) is equivalent to f(f(1,2),3)
#' 
## ------------------------------------------------------------------------
set.seed(32611)
l <- replicate(4, sample(1:10, 15, replace = T), simplify = FALSE)
str(l)
intersect(intersect(intersect(l[[1]], l[[2]]),
  l[[3]]), l[[4]])
Reduce(intersect,l)

#' 
#' 
#' Reduce 2
#' ===
#' 
## ------------------------------------------------------------------------
a <- c(1:10)
Reduce('+', a)

b = as.list(letters[1:10])
Reduce("paste0", b)

d <- as.list(letters[1:10])
Reduce("c",d)

#' 
#' Reduce 3
#' ===
#' 
## ------------------------------------------------------------------------
## Iterative function application:
Funcall <- function(f, ...) f(...)
## Compute log(exp(acos(cos(0))
Reduce(Funcall, list(log, exp, acos, cos), 0, right = TRUE)

#' 
#' Filter
#' ===
#' 
## ------------------------------------------------------------------------
alist = list(a="a", b=1, c=FALSE, d=list(1:10), e="e", f = 2)
Filter(is.character, alist)

#' 
#' Find
#' ===
## ------------------------------------------------------------------------
alist = list(a="a", b=1, c=FALSE, d=list(1:10), e="e", f = 2)
Find(is.character, alist)
Find(is.numeric, alist)

#' 
#' Position
#' ===
## ------------------------------------------------------------------------
alist = list(a="a", b=1, c=FALSE, d=list(1:10), e="e", f = 2)
Position(is.character, alist)
Position(is.numeric, alist)

#' 
#' 
#' 
#' Outer
#' ===
#' 
#' - It takes multiple vector inputs and creates a matrix or array output where the input function is run over every combination of the inputs.
## ------------------------------------------------------------------------
outer(1:3, 1:5, "*")
1:3 %o% 1:5

#' 
#' Outer 2
#' ===
## ------------------------------------------------------------------------
outer(1:3, 1:5, "paste")

outer(1:3, 1:5, ">")

outer(1:3, 1:5, "^")

#' 
#' 
#' Vectorize a function
#' ===
#' - if your function has to take a scaler as input, you can vectorize it by vectorize function
## ------------------------------------------------------------------------
combn(4,2) ## all combinations of "choose 2 numbers from 1:4"
## However, you cannot do combn(4:5,2:3), how to vectorize this function

combnV <- Vectorize(function(x, m) combn(x, m),
                    vectorize.args = c("x", "m"))

combnV(4:5,2:3)

#' 
#' 
#' Balance between efficiency and simplicity of your code
#' ===
#' - If looping is very time-consuming relative to the actual operation, you may need to consider vectorized calculation to improve efficiency.
#' - If looping is ignorable comparing to the actual operation (e.g. Gibbs sampling), you can keep your loop to make it readible.
#' 
#' 
## ----echo=FALSE----------------------------------------------------------
suppressWarnings(library(knitr))
purl("vectorized.rmd", output = "vectorized.R ", documentation = 2)

