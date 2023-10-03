#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Tuesday Oct 3rd, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Rcpp"
#' ---
#' 
#' 
#'  
#' Why we need C++ in R
#' ===
#' - Sometimes R is not efficient, it is user friendly but not process friendly.
#' - R is extreme inefficient for loop, but C++ is very efficient.
#' - Problems that require advanced data structures and algorithms that R doesn't provide. C++ has efficient implementations of many important data structures.
#' 
#' 
#' Don't know C++? Not a problem.
#' ===
#' 
#' - A working knowledge of C++ is helpful, but not essential.
#' - Many good tutorials and references are freely available, including:
#'     - http://www.learncpp.com/
#'     - http://www.cplusplus.com/
#'     - Book: Seamless R and C++ Integration with Rcpp, (Dirk Eddelbuettel)
#' 
#' 
#' Rcpp
#' ===
#' 
#' Rcpp: for Seamless R and C++ Integration.
#' 
#' - The Rcpp package has become the most widely used language extension for R.
#' - As of July 2023, 2712 packages on CRAN and a further 250 on
#' BioConductor deploy Rcpp to extend R.
#' 
#' 
#' 
#' Rcpp Prerequisites:
#' ===
#' 
#' - Install the latest version of Rcpp from CRAN with install.packages("Rcpp").
#' - The package includes cppFunction() and sourceCpp(), which makes it very easy to connect C++ to R.
#' - You'll also need a working C++ compiler. To get it:
#'     - On Windows, install Rtools.
#'     - On Mac, install Xcode from the app store.
#'     - On Linux, sudo apt-get install r-base-dev or similar.
#' 
#' 
#' Warm up example
#' ===
#' 
#' cppFunction() allows you to write C++ functions in R.
#' 
## -----------------------------------------------------------------
## Getting started with C++
library(Rcpp)
cppFunction('int add(int x, int y) {
  int sum = x + y;
  return sum;
  }')
# add works like a regular R function
add
add(3,2)

#' 
#' 
#' Note on C++ code
#' ===
#' - In C++, you have to declare types of input and output.
#' - (C++ specific) The scalar equivalents:  
#'   - double: double
#'   - integer: int
#'   - character: string
#'   - logical: bool
#' - (Rcpp specific) The classes for the most common types of R vectors are:
#'   - NumericVector
#'   - IntegerVector
#'   - CharacterVector
#'   - LogicalVector.
#' - You must use an explicit return statement to return a value from a function.
#' - **Every statement is terminated by a ";".**
#' 
#' 
#' More examples
#' ===
#' 1. No inputs, scalar output.
#' 2. Scalar input, scalar output.
#' 3. Vector input, scalar output.
#' 4. Vector input, vector output.
#' 5. Matrix input, vector output.
#' 
#' No inputs, scalar output (1).
#' ===
#' 
#' - in R
## -----------------------------------------------------------------
one <- function() 1L

#' 
#' - in C
## -----------------------------------------------------------------
cppFunction('int oneC() {
              return 1;
            }')

#' 
#' - evaluation
## -----------------------------------------------------------------
one()
oneC()

#' 
#' 
#' Scalar input, scalar output (2).
#' ===
#' 
#' - in R
## -----------------------------------------------------------------
signR <- function(x) {
  if (x > 0) {
    1
  } else if (x == 0) {
    0
  } else {
    -1
  }
}

#' 
#' - in C
## -----------------------------------------------------------------
cppFunction('int signC(int x) {
              if (x > 0) {
                return 1;
              } else if (x == 0) {
                return 0;
              } else {
                return -1;
              }
            }')

#' 
#' - evaluation
## -----------------------------------------------------------------
signR(-10)
signC(-10)

#' 
#' 
#' Vector input, scalar output (3).
#' ===
#' - in R
## -----------------------------------------------------------------
sumR <- function(x) {
  total <- 0
  for (i in seq_along(x)) {
    total <- total + x[i]
  }
  total
}

#' - in C
## -----------------------------------------------------------------
cppFunction('double sumC(NumericVector x) {
              int n = x.size();
              double total = 0;
              for(int i = 0; i < n; ++i) {
                total += x[i];
              }
              return total;
            }')

#' 
#' Vector input, scalar output (3) - evaluation
#' ===
## -----------------------------------------------------------------
x <- runif(1e7)
system.time(sum(x))
system.time(sumC(x))
system.time(sumR(x))

#' 
#' Vector input, vector output (4).
#' ===
#' 
#' - in R
## -----------------------------------------------------------------
pdistR <- function(x, ys) {
  sqrt((x - ys) ^ 2)
}

#' 
#' - in C
## -----------------------------------------------------------------
cppFunction('NumericVector pdistC(double x, NumericVector ys) {
              int n = ys.size();
              NumericVector out(n);
               
              for(int i = 0; i < n; ++i) {
                out[i] = sqrt(pow(ys[i] - x, 2.0));
              }
              return out;
            }')

#' 
#' - evaluation
## -----------------------------------------------------------------
pdistR(3,1:5)
pdistC(3,1:5)

#' 
#' 
#' Matrix input, vector output (5).
#' ===
#' 
#' - in R
## ---- eval=FALSE--------------------------------------------------
## rowSums

#' 
#' - in C
## -----------------------------------------------------------------
cppFunction('NumericVector rowSumsC(NumericMatrix x) {
              int nrow = x.nrow(), ncol = x.ncol();
              NumericVector out(nrow);
            
              for (int i = 0; i < nrow; i++) {
                double total = 0;
                for (int j = 0; j < ncol; j++) {
                  total += x(i, j);
                }
                out[i] = total;
              }
              return out;
            }')

#' 
#' - evaluation
## -----------------------------------------------------------------
set.seed(1014)
x <- matrix(sample(100), 10)
rowSums(x)
rowSumsC(x)

#' 
#' 
#' 
#' 
#' More note on C++ code
#' ===
#' - Syntax **if**, **while**, **break** are identical, but to skip one iteration you need to use **continue** instead of **next**.
#' - To find the length of the vector, we use the .size() method, which returns an integer. (C++ is objective oriented language, . is similar to slot in S4 in R).
#' - The for statement has a different syntax: for(init; check; increment).
#' - This loop is initialized by creating a new variable called i with value 0.
#' Before each iteration we check that i < n, and terminate the loop if it's not.
#' - In C++, vector indices start at 0. I'll say this again because it's so important: **IN C++, VECTOR INDICES START AT 0!**
#' - Use = for assignment, not <-.
#' - C++ provides operators that modify in-place: 
#'   - total += x[i] is equivalent to total = total + x[i]. 
#'   - Similar in-place operators are -=, *=, and /=.
#' 
#' Using sourceCpp
#' ===
#' - So far, we've used inline C++ with cppFunction().
#' - For real problems, it's usually easier to use stand-alone C++ files and then source them into R using sourceCpp().
#' - Your stand-alone C++ file should have extension .cpp, and needs to
#' start with:
#' 
## #include <Rcpp.h>

## using namespace Rcpp;

## // [[Rcpp::export]]

#' 
#' - To compile the C++ code, use sourceCpp("path/to/file.cpp").
#' - See examples: meanC.cpp
#' 
#' 
#' Using sourceCpp
#' ===
#' 
#' - [meanC.cpp](https://caleb-huo.github.io/teaching/2017FALL/lectures/week8_RPackage/RCPP/src/meanC.cpp)
#' 
## #include <Rcpp.h>

## using namespace Rcpp;

## 

## // [[Rcpp::export]]

## double meanC(NumericVector x) {

##   int n = x.size();

##   double total = 0;

## 

##   for(int i = 0; i < n; ++i) {

##     total += x[i];

##   }

##   return total/n;

## }

#' 
#' - In R:
## ---- eval = F----------------------------------------------------
## sourceCpp('meanC.cpp')
## meanC(1:100)

#' 
#' list - mean percentage error
#' ===
#' 
#' - [mpe.cpp](https://caleb-huo.github.io/teaching/2017FALL/lectures/week8_RPackage/RCPP/src/mpe.cpp)
#' 
## #include <Rcpp.h>

## using namespace Rcpp;

## 

## // [[Rcpp::export]]

## double mpe(List mod) {

##   if (!mod.inherits("lm")) stop("Input must be a linear model");

## 

##   NumericVector resid = as<NumericVector>(mod["residuals"]);

##   NumericVector fitted = as<NumericVector>(mod["fitted.values"]);

## 

##   int n = resid.size();

##   double err = 0;

##   for(int i = 0; i < n; ++i) {

##     err += resid[i] / (fitted[i] + resid[i]);

##   }

##   return err / n;

## }

#' 
#' list - mean percentage error
#' ===
#' 
## ---- eval=F------------------------------------------------------
## sourceCpp('mpe.cpp')
## mod <- lm(mpg ~ wt, data = mtcars)
## mpe(mod)

#' 
#' 
#' lapply
#' ===
#' 
#' - [lapply1.cpp](https://caleb-huo.github.io/teaching/2017FALL/lectures/week8_RPackage/RCPP/src/lapply1.cpp)
#' 
## #include <Rcpp.h>

## using namespace Rcpp;

## 

## // [[Rcpp::export]]

## List lapply1(List input, Function f) {

##   int n = input.size();

##   List out(n);

## 

##   for(int i = 0; i < n; i++) {

##     out[i] = f(input[i]);

##   }

## 

##   return out;

## }

#' 
## ---- eval = F----------------------------------------------------
## sourceCpp('lapply1.cpp')
## 
## lapply(1:10,function(x) x^2)
## 
## lapply1(1:10,function(x) x^2)
## 

#' 
#' 
#' Gibbs sampling example
#' ===
#' 
#' - <http://dirk.eddelbuettel.com/blog/2011/07/14/>
#'     - Consider the same Gibbs sampler for a bivariate distribution:
#'         $$f(x,y) = k x^2 \exp( -x y^2 - y^2 + 2y - 4x)$$
#'     - conditional distributions are:
#'         $$f(x|y) = (x^2)* \exp(-x*(4+y*y)),$$ which is a Gamma density. 
#'         $$f(y|x) = \exp(-0.5*2*(x+1)*(y^2 - 2*y/(x+1)),$$ which is a Gaussian density.
#' 
#' 
#' 
#' 
## -----------------------------------------------------------------
gibbs_r <- function(N, thin) {
  mat <- matrix(nrow = N, ncol = 2)
  x <- y <- 0

  for (i in 1:N) {
    for (j in 1:thin) {
      x <- rgamma(1, 3, y * y + 4)
      y <- rnorm(1, 1 / (x + 1), 1 / sqrt(2 * (x + 1)))
    }
    mat[i, ] <- c(x, y)
  }
  mat
}

#' 
#' 
#' 
#' Gibbs sampling evaluation
#' ===
#' 
#' - [gibbs_cpp.cpp](https://caleb-huo.github.io/teaching/2017FALL/lectures/week8_RPackage/RCPP/src/gibbs_cpp.cpp)
#' 
## ---- eval = F----------------------------------------------------
## sourceCpp('gibbs_cpp.cpp')
## 
## system.time(gibbs_r(100, 10))
## system.time(gibbs_cpp(100, 10))

#' 
#' 
#' Using Rcpp in a package (minimum steps)
#' ===
#' 
#' Also refer to <https://caleb-huo.github.io/teaching/2023FALL/lectures/Week7_RPackage/Rpackage.html> for regular steps
#' 
#' 1. usethis::create_package() create an new R package
#' 2. devtools::document() 
#' 3. usethis::use_rcpp()
#' 4. Copy your R code with documentation in inside R folder
#' 5. put cpp file in src
#' 6. Rcpp::compileAttributes()
#' 8. devtools::document() generate R documentation
#' 9. devtools::install() install the package
#' 
#' 
#' 
#' example r file
#' ===
#' 
#' If your package name is GatorCPP
#' 
#' - need to put the following somewhere in the .R file, following the instruction of Rcpp::compileAttributes()
#'   - ##' @useDynLib GatorCPP 
#'   - ##' @importFrom Rcpp sourceCpp
#' 
#' - the newer version of R/RCPP may generate different instructions, suggest to use the instruction in the slides.
#' 
## ---- eval = F----------------------------------------------------
## ##' mean calling c
## ##'
## ##' mean calling c balalal
## ##' @title mean calling c
## ##' @param x a vector
## ##' @return mean of the vector
## ##' @author Caleb
## ##' @export
## ##' @useDynLib GatorCPP
## ##' @importFrom Rcpp sourceCpp
## ##' @examples
## ##' meanRC(1:10)
## meanRC <- function(x){
##   meanC(x)
## }

#' 
#' 
#' example c file
#' ===
#' 
#' 
## #include <Rcpp.h>

## using namespace Rcpp;

## 

## // [[Rcpp::export]]

## double meanC(NumericVector x) {

##   int n = x.size();

##   double total = 0;

## 

##   for(int i = 0; i < n; ++i) {

##     total += x[i];

##   }

##   return total / n;

## }

#' 
#' Wrap up the package
#' ===
#' 
#' 
## ---- eval = F----------------------------------------------------
## WD <- "~/Desktop"
## setwd(WD)
## usethis::create_package("GatorCPP",open = FALSE) ## create the package
## setwd(file.path(WD, "GatorCPP")) ## get into the package
## devtools::document() ## need to do this, otherwise the next step use_rcpp won't work
## usethis::use_rcpp() ## intialize rcpp
## ## put in R code in R and cpp code in src
## Rcpp::compileAttributes() ## will generate a RcppExports.R
## # devtools::load_all() ## this step seems not necessary this year
## devtools::document() ## generate documentation
## devtools::install() ## install the package
## ## use the package
## library(GatorCPP)
## #?meanRC
## meanRC(1:10)

#' 
#' 
#' 
#' 
#' Reference
#' ===
#' 
#' - http://adv-r.had.co.nz/Rcpp.html
#' - https://cran.r-project.org/web/packages/Rcpp/vignettes/Rcpp-package.pdf
#' 
#' 
