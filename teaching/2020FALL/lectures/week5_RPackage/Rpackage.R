#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "R package"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Sep 28, 2020"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' 
#' Install R packages
#' ===
#' - from CRAN, 14,925 packages, up to 9/17/2019.
## ---- eval = F-----------------------------------------------------------
## install.packages("mclust")

#' - from Bioconductor: 1,741 packages, 9/17/2019.
## ---- eval=F-------------------------------------------------------------
## ## try http:// if https:// URLs are not supported
## if (!requireNamespace("BiocManager", quietly = TRUE))
##     install.packages("BiocManager")
## 
## BiocManager::install("impute")

#' - .gz.tar zipped file.
## ---- eval=F-------------------------------------------------------------
## install.packages("mclust_5.3.tar",repos=NULL,type="source")

#' - package folder.
## ---- eval=F-------------------------------------------------------------
## # in R:
## system("R CMD INSTALL mclust")
## #command line: R CMD INSTALL mclust

#' - github.
## ---- eval=F-------------------------------------------------------------
## library(devtools)
## install_github("cran/mclust")

#' 
#' 
#' Why R package
#' ===
#' - Share with others, if your code is in a package, any R user can easily download it, install it and learn how to use it.
#' - It is about saving yourself time. Organizing code in a package makes your life easier because packages come with conventions.
#' 
#' 
#' 
#' 
#' 
#' Basic structure of an R Package 
#' ===
#' 
#' 
#' ![](../figure/packageStructure.png)
#' 
#' - Using R survival package as an example:
#'     - <span style="color:blue">DESCRIPTION</span>: metadata, basic information, dependency...
#'     - <span style="color:blue">NAMESPACE</span>: control the imported and exported objects.
#'     - <span style="color:blue">R</span>: R code 
#'     - <span style="color:blue">man</span>: manual 
#'     - src: other language (c, java...)
#'     - test: test validity of your functions
#'     - vignettes: an overall description of your package
#'     - ToDo: to do list
#' 
#' Play with survival package
#' ===
#' - library a package
## ---- eval=F-------------------------------------------------------------
## library(survival)

#' - help file about the package
## ---- eval=F-------------------------------------------------------------
## help(package = 'survival')

#' - help file about one function
## ---- eval=F-------------------------------------------------------------
## ?coxph

#' - check out vignettes
## ---- eval=F-------------------------------------------------------------
## browseVignettes("survival")

#' 
#' 
#' Structure of R package
#' ===
#' - R/: the most important directory is R/
#' - DESCRIPTION: basic information, dependency.
#' - Documentation: if you want other people (including future-you!) to understand how to use the functions in your package, you'll need to document them.
#' - Vignettes: function documentation describes the nit-picky details of every function in your package.They're long-form documents that show how to combine multiple parts of your package to solve real problems.
#' - Tests: to ensure your package works as designed (and continues to
#' work as you make changes), it's essential to write unit tests which
#' define correct behaviour, and alert you when functions break.
#' - Namespace: to play nicely with others, your package needs to define what functions it makes available to other packages and what functions it requires from other packages.
#' - data/: this directory allows you to include data with your package.
#' - src/: other programming languages.
#' 
#' 
#' 
#' 
#' DESCRIPTION
#' ===
#' The job of the DESCRIPTION file is to store important metadata about your package. Every package must have a DESCRIPTION. In fact, it's the defining feature of a package.
#'   
#' - LazyLoad, if yes delays the loading of functions until they are required.
#' - Title and description.
#' - Author: who are you.
#' - Dependencies.
#' - License.
#' - Version.  
#' 
#' 
#' DESCRIPTION License
#' ===
#' 
#' - GPL-2/GPL-3: anyone who distributes modified versions of your code (derivative works) must also make the source code 
#'   - open source, 
#'   - GPL-compatible.
#' - MIT: This is a simple and permissive license. 
#'   - They are extremely short and essentially say “do whatever you want with this, just don’t sue me.”
#' - CC0. It relinquishes all your rights on the code and data so that it can be freely used by anyone for any purpose.
#' - Apache:
#'   - “do whatever you want with this, just don’t sue me” but does so with many more words
#'   - Contains patent license and retaliation clause to prevent patents from encumbering the software project
#' - Many others
#' 
#' DESCRIPTION Version
#' ===
#' Formally, an R package version is a sequence of at least two integers separated by either . or -. For example, 1.0 and 0.9.1-10 are valid versions, but 1 or 1.0-devel are not.
#' 
#' - A released version number consists of three numbers, <major>.<minor>.<patch>. For version number 1.9.2, 1 is the major number, 9 is the minor number, and 2 is the patch number. Never use versions like 1.0, instead always spell out the three components, 1.0.0.
#' 
#' - An in-development package has a fourth component: the development version. This should start at 9000. For example, the first version of the package should be 0.0.0.9000.
#' 
#' The namespace file in R package
#' ===
#' 
#' 
#' - Specifies which functions will be visible to users.
#' - Specifies which functions are internal funciton, for which you don't want the users to see and use.
#' - Specifies how to interaction with other programming language (e.g., c++)
#' - Specifies how to your package depends on other packages
#' - Don't need to revise the namespace file, will be managed by devtools.
#' 
#' Namespace
#' ===
#' 
#' - :: to specify namespace
#' ```
#' base::dim
#' ```
#' 
#' - Namespace protect the internal functions being overwritten by functions from other packages/global environment.
#' 
#' ```
#' nrow
#' dim(mtcars)
#' dim <- function(x) c(1, 1)
#' dim(mtcars)
#' nrow(mtcars)
#' ```
#' 
#' Man (for manual)
#' ===
#' 
#' - Documentation is one of the most important aspects of a good
#' package. Without it, users won't know how to use your package.
#' - Documentation is also useful for future-you (so you remember what
#' your functions were supposed to do), and for developers extending
#' your package.
#' - R provides a standard way of documenting the objects in a package:
#' you write .Rd files in the man/ directory. These files use a custom
#' syntax, loosely based on LaTeX, and are rendered to HTML, plain
#' text and pdf for viewing.
#' 
#' 
#' 
#' 
#' 
#' Devtools, starting from sketches. 
#' ===
#' 
#' <span style="color:red">Red color indicates essential steps</span>
#' 
#' 1. <span style="color:red">usethis::create_package() create an new R package</span>
#' 2. <span style="color:red">Copy your R code in inside R folder</span>
#' 3. formatR::tidy_dir() clean up the code
#' 4. <span style="color:blue">Add documentation in R code</span>
#' 5. <span style="color:red">devtools::document() generate R documentation</span>
#' 6. usethis::use_data() prepare external data
#' 7. usethis::use_testthat() prepare test functions
#' 8. devtools::test() preform test
#' 9. usethis::use_vignette() generate vignettes
#' 10. devtools::check() check the package
#' 11. devtools::build() build the package
#' 12. <span style="color:red">devtools::install() install the package</span>
#' 13. others
#' 
#' 
#' <span style="color:red">1. devtools::create() create an new R package</span>
#' ===
#' 
## ---- eval=F-------------------------------------------------------------
## ## set working directory to be Desktop
## WD <- '~/Desktop'
## setwd(WD)
## 
## usethis::create_package("GatorPKG", open = FALSE) ## open = FALSE will prevent R open a new R studio session.
## WD2 <- '~/Desktop/GatorPKG'
## setwd(WD2)

#' 
#' <span style="color:red">2. Copy your R code in inside R folder</span>
#' ===
#' 
#' - f.R
#' 
## ---- eval=F-------------------------------------------------------------
## ##' @export
## f <- function(x, y) x + y

#' 
#' - g.R
#' 
## ---- eval=F-------------------------------------------------------------
## ##' @export
## g <- function(x, y) x - y

#' 
#' - h.R
#' 
## ---- eval=F-------------------------------------------------------------
## ##' @export
## h <- function(x, y) f(x,y) * g(x,y)

#' 
#' - all.R
## ---- eval=F-------------------------------------------------------------
## ##' @export
## all <- function(x, y){
##   f0 <- f(x,y)
##   g0 <- g(x,y)
##   h0 <- h(x,y)
##   list(f=f0, g=g0, h=h0)
## }

#' 
#' 
#' 3. formatR::tidy dir() clean up the code
#' ===
#' 
## ---- eval = F-----------------------------------------------------------
## setwd(WD2)
## ## make the code neat
## formatR::tidy_dir("R")

#' 
#' <span style="color:blue">4. Add documentation in R code </span>
#' ===
#' 
#' - f.R
## ---- eval = F-----------------------------------------------------------
## ##' Add up two numbers (Description)
## ##'
## ##' We want to add up two numbers, blalala... (Details)
## ##' @title add two numbers
## ##' @param x first number
## ##' @param y second number
## ##' @return sum of two numbers
## ##' @author Caleb
## ##' @export
## ##' @examples
## ##' f(1,2)
## f <- function(x, y) x + y

#' 
#' ---
#' 
#' - g.R
## ---- eval = F-----------------------------------------------------------
## ##' Subtract two numbers (Description)
## ##'
## ##' We want to Subtract two numbers, blalala... (Details)
## ##' @title Subtract two numbers
## ##' @param x first number
## ##' @param y second number
## ##' @return x - y
## ##' @author Caleb
## ##' @export
## ##' @examples
## ##' g(2,1)
## g <- function(x, y) x - y

#' 
#' ---
#' 
#' - h.R
## ---- eval = F-----------------------------------------------------------
## ##' Complex operations on two numbers (Description)
## ##'
## ##' We want to do some complex operations on two numbers, blalala... (Details)
## ##' @title Very complex operation of two numbers
## ##' @param x first number
## ##' @param y second number
## ##' @return (x - y)(x + y)
## ##' @author Caleb
## ##' @export
## ##' @examples
## ##' h(3,2)
## h <- function(x, y) f(x,y) * g(x,y)

#' 
#' ---
#' 
#' - all.R
## ---- eval = F-----------------------------------------------------------
## ##' Return f,g,h (Description)
## ##'
## ##' We want to return f,g,h , blalala... (Details)
## ##' @title return all
## ##' @param x first number
## ##' @param y second number
## ##' @return A list of f, g, and h.
## ##' \item{f}{Results for adding}
## ##' \item{g}{Results for subtracting}
## ##' \item{h}{Complex Result}
## ##' @author Caleb
## ##' @export
## ##' @examples
## ##' all(3,2)
## all <- function(x, y){
##   f0 <- f(x,y)
##   g0 <- g(x,y)
##   h0 <- h(x,y)
##   list(f=f0, g=g0, h=h0)
## }

#' 
#' <span style="color:red">5. devtools::document() generate R documentation</span>
#' ===
#' 
## ---- eval=F-------------------------------------------------------------
## devtools::document() ## default argument is pkg = ".", current working directory

#' 
#' Change:
#' 
#' - NAMESPACE is updated
#' - man folder is created
#' 
#' 6. usethis::use_data() prepare external data
#' ===
#' 
## ---- eval=F-------------------------------------------------------------
## xxxx <- sample(1000)
## usethis::use_data(xxxx)

#' 
#' Change:
#' 
#' - data folder
#' - after install the package, do the following:
#' 
#' ```
#' data(xxxx)
#' ```
#' 
#' 6. documenting external data
#' ===
#' 
#' - no need to export external data
#' 
#' xxxx.R
## ---- eval=F-------------------------------------------------------------
## #' Prices of 50,000 round cut diamonds.
## #'
## #' A dataset containing the prices and other attributes of almost 54,000
## #' diamonds.
## #'
## #' @format A data frame with 53940 rows and 10 variables:
## #' \describe{
## #'   \item{price}{price, in US dollars}
## #'   \item{carat}{weight of the diamond, in carats}
## #'   ...
## #' }
## #' @source \url{http://www.diamondse.info/}
## "xxxx"

#' 
#' 
#' 6. internal data
#' ===
#' 
#' - Internal data: These objects are only available within the package namespace.
#' - Will be saved as sysdata.rda in R/
#' - No documentation
#' - Good to put the raw code in data-raw/ for reproducibility purpose
#' 
## ---- eval=F-------------------------------------------------------------
## yyyy <- sample(1000)
## usethis::use_data(yyyy, internal = TRUE)

#' 
#' ---
#' 
#' - printY.R
#' 
## ---- eval = F-----------------------------------------------------------
## ##' @export
## printY <- function(){
##   print(yyyy)
## }

#' 
#' 
#' 7. usethis::use_testthat() prepare test functions
#' ===
#' 
#' - create test functions
## ---- eval = F-----------------------------------------------------------
## usethis::use_testthat() ## default argument is pkg = ".", current working directory

#' 
#' - actual test function (e.g. testthat/testf.R)
## ---- eval = F-----------------------------------------------------------
## test_that("test if f function is correct", {
##     expect_equal(f(1,1), 2)
##   }
## )
## 
## test_that("test if f function is correct", {
##     expect_equal(f(1,4), 2)
##   }
## )

#' 
#' 8. devtools::test() preform test
#' ===
#' 
#' - perform test
## ---- eval = F-----------------------------------------------------------
## devtools::test() ## default argument is pkg = ".", current working directory

#' 
#' 
#' 
#' 9. usethis::use_vignette() generate vignettes
#' ===
#' 
#' - browse all Vignettes
## ---- eval = F-----------------------------------------------------------
## browseVignettes()

#' 
#' - create our own Vignettes
#' 
## ---- eval=F-------------------------------------------------------------
## setwd(WD2)
## usethis::use_vignette("Gators")

#' 
#' - Turn build_vignettes on when install the package
#' 
## ---- eval = FALSE-------------------------------------------------------
## devtools::install(build_vignettes = TRUE)

#' 
#' 
#' 10. devtools::check() check the package 
#' ===
#' 
#' - check the package
## ---- eval = F-----------------------------------------------------------
## setwd(WD2)
## devtools::check() ## default argument is pkg = ".", current working directory

#' 
#' 11. devtools::build() build the package
#' ===
#' 
#' - build() will generate .tar.gz package
#' 
## ---- eval=F-------------------------------------------------------------
## ## build the package
## devtools::build() ## default argument is pkg = ".", current working directory

#' 
#' - an inst folder will be created, which contains your vignettes.
#' 
#' 
#' <span style="color:red">12. devtools::install() install the package</span>
#' ===
#' 
#' - install the package
## ---- eval=F-------------------------------------------------------------
## devtools::install() ## default argument is pkg = ".", current working directory
## devtools::install(build_vignettes = TRUE) ## also build the vignettes

#' 
#' - remove the package
## ---- eval=F-------------------------------------------------------------
## remove.packages("GatorPKG")

#' 
#' - install the pacakge again
## ---- eval=F-------------------------------------------------------------
## install.packages(file.path("~/Desktop","GatorPKG_0.0.0.9000.tar.gz"),repos=NULL,type="source") ## directly with vignettes

#' 
#' 
#' 13, Package checks
#' ===
#' 
#' - R CMD check 
#' 
## ---- eval = FALSE-------------------------------------------------------
## system("R CMD check ~/Desktop/GatorPKG_0.0.0.9000.tar.gz")

#' 
#' This is equivalent to running the following in linux.
#' 
#' ```
#' R CMD check ~/Desktop/GatorPKG_0.0.0.9000.tar.gz
#' ```
#' 
#' 14, others (internal function)
#' ===
#' 
#' - internal function: you want to use internally but don't want other people to see it.
#' - If we do not use ##` @export, the function will become a internal function
#' 
#' 15a, others (depend on other packages)
#' ===
#' 
#' - DESCRIPTION file
#' Depends: R (>= 3.6.0), survival
#' 
#' ```
#' Package: GatorPKG
#' Title: What the Package Does (one line, title case)
#' Version: 0.0.0.9000
#' Authors@R: person("First", "Last", email = "first.last@example.com", role = c("aut", "cre"))
#' Description: What the package does (one paragraph).
#' Depends: R (>= 3.6.0), survival
#' License: What license is it under?
#' Encoding: UTF-8
#' LazyData: true
#' RoxygenNote: 6.1.1
#' ```
#' 
#' ---
#' 
#' - change your function f.R to depend on coxph
#' 
## ----eval = FALSE--------------------------------------------------------
## ##' Add up two numbers (Description)
## ##'
## ##' We want to add up two numbers, blalala... (Details)
## ##' @title add two numbers
## ##' @param x first number
## ##' @param y second number
## ##' @return sum of two numbers
## ##' @author Caleb
## ##' @export
## ##' @examples
## ##' f(1,2)
## f <- function(x, y){
##   print(head(coxph))	
##   x + y	
## }

#' 
#' - all visible functions in the survival package will be exported in your package.
#' 
#' 15b, others (depend on other packages)
#' ===
#' 
#' What if you do not want to load the entire dependent package, but only certain functions
#' 
#' - Use @import in your function
#' - Don't need to declare dependency in DESCRIPTION
#' 
## ----eval = FALSE--------------------------------------------------------
## ##' Add up two numbers (Description)
## ##'
## ##' We want to add up two numbers, blalala... (Details)
## ##' @title add two numbers
## ##' @param x first number
## ##' @param y second number
## ##' @return sum of two numbers
## ##' @author Caleb
## ##' @import survival
## ##' @export
## ##' @examples
## ##' f(1,2)
## f <- function(x, y){
##   print(head(coxph))	
##   x + y	
## }

#' 
#' 
#' ---
#' 
#' - update namespace
## ---- eval = FALSE-------------------------------------------------------
## devtools::document()

#' 
#' - reinstall the package
#' 
#' Your function depends on the external package, but the external package is not loaded.
#' 
#' 
#' 16, others (R package and GitHub)
#' ===
#' 
#' - GitHub is a great open platform to host R packages
#'    - Apply for a github acccount
#'    - https://github.com
#'    - Free GitHub pro account (GitHub Education): requires scanning of the UF ID card.
#'    
#' - Play with the following git tutorials (HW)
#'   - https://www.codecademy.com/learn/learn-git
#' 
#' 
#' References
#' ===
#' 
#' - [http://adv-r.had.co.nz](http://adv-r.had.co.nz)
#' - https://exygy.com/which-license-should-i-use-mit-vs-apache-vs-gpl/
#' 
