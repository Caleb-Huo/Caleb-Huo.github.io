##' mean calling c
##'
##' mean calling c balalal
##' @title mean calling c
##' @param x a vector
##' @return mean of the vector
##' @author Caleb
##' @export
##' @useDynLib GatorCPP
##' @importFrom Rcpp sourceCpp
##' @examples
##' meanRC(1:10)
meanRC <- function(x){
  meanC(x)
}
