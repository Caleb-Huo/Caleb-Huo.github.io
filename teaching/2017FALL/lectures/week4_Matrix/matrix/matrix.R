#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday September 20, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Matrix operation
#' ---
#' 
#' Summary of all matrix operations
#' ===
#' <http://www.statmethods.net/advstats/matrix.html>
#' 
#' Create a matrix
#' ===
#' - an example from ?matrix
## ------------------------------------------------------------------------
mdat <- matrix(c(1,2,3, 11,12,13), nrow = 2, ncol = 3, byrow = FALSE,
               dimnames = list(c("row1", "row2"),
                               c("C.1", "C.2", "C.3")))
mdat

#' 
#' - an alternative way
## ------------------------------------------------------------------------
mdat <- c(1,2,3, 11,12,13)
dim(mdat) <- c(2,3)
dimnames(mdat) <- list(c("row1", "row2"), c("C.1", "C.2", "C.3"))
mdat

#' 
#' Appending a column  or a row to a matrix
#' ===
## ------------------------------------------------------------------------
dim(mdat)

#' - appending a column
## ------------------------------------------------------------------------
mat2 <- cbind(mdat, c(3,4))
dim(mat2)

#' 
#' - appending a row
## ------------------------------------------------------------------------
mat3 <- rbind(mdat, c(2,5,8))
dim(mat3)

#' 
#' 
#' Subset of a matrix
#' ===
#' - by numeric index
## ------------------------------------------------------------------------
mdat[1,1]
mdat[1,1:2]
mdat[1,]
mdat[1,-1]
mdat[,]

#' 
#' ---
#' 
#' - by names
## ------------------------------------------------------------------------
dimnames(mdat); rownames(mdat); colnames(mdat)
mdat["row1", "C.1"]
mdat["row2", c("C.2", "C.3")]
mdat["row1",-2] ## mixed usage

#' 
#' Transponse of a Matrix
#' ===
#' - t()
## ------------------------------------------------------------------------
mat1 <- matrix(1:6,nrow=2,ncol=3)
mat1
t(mat1)

#' 
#' Element-wise Multiplication by a Scalar/Vector/Matrix
#' ===
#' - by a scalar
## ------------------------------------------------------------------------
mdat * 3

#' 
#' - by a **column** vector
## ------------------------------------------------------------------------
mdat * c(1,2)

#' 
#' 
#' - by a matrix of the same dimension
## ------------------------------------------------------------------------
mdat * matrix(1:6,nrow=2,ncol=3)

#' 
#' 
#' 
#' Addition/Subtraction by a Scalar/Vector/Matrix
#' ===
#' - by a scalar
## ------------------------------------------------------------------------
mdat + 3

#' 
#' - by a **column** vector
## ------------------------------------------------------------------------
mdat - c(1,2)

#' 
#' 
#' - by a matrix of the same dimension
## ------------------------------------------------------------------------
mdat + matrix(1:6,nrow=2,ncol=3)

#' 
#' 
#' 
#' Matrix Multiplication
#' ===
#' - %*%
#' 
## ------------------------------------------------------------------------
mat1 <- matrix(1:6,nrow=2,ncol=3)
mat1

mat2 <- matrix(1:6,nrow=3,ncol=2)
mat2

mat1 %*% mat2

mat2 %*% mat1

#' 
#' crossProd
#' ===
#' 
#' - crossprod(A,B): $A^\top B$
#' - crossprod(A)	$A^\top A$
#' 
## ------------------------------------------------------------------------
A <- matrix(c(1,2), ncol =1)
B <- matrix(c(2,2), ncol=1)

A 
B

crossprod(A,B)
crossprod(t(A),t(B))
crossprod(A)
crossprod(t(A))

#' 
#' 
#' Diagonal Matrix
#' ===
#' - diag()
#'     - if input is a square matrix, return the diagnal element.
#'     - if input is a vector, return a diagnal matrix.
#'     - if input is a scalar (i.e. k), return this creates a k x k identity matrix
## ------------------------------------------------------------------------
S <- matrix(1:9,3,3)
S
diag(S)
diag(c(1,4,7))
diag(3)

#' 
#' 
#' Inverse of a Matrix
#' ===
## ------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
SI <- solve(S)
SI
S %*% SI
SI %*% S


#' 
#' More about solve
#' ===
#' - solve(A, b)	Returns vector x in the equation $b = Ax$ (i.e., $A^{-1}b$)
#' 
## ------------------------------------------------------------------------
set.seed(32608)
(A <- matrix(rnorm(9),3,3))
(b <- runif(3))
(x <- solve(A,b))

A %*% x

#' 
#' 
#' Moore-Penrose Generalized Inverse
#' ===
#' 
#' - Generalized inverse of a matrix $A\in \mathbb{R}^{n\times m}$ is defined as $A^g \in \mathbb{R}^{m\times n}$:
#' $$AA^gA = A$$
#' - Regular inverse is a special case of Generalized Inverse when $m = n$ and $A$ is non-singular.
#' 
## ------------------------------------------------------------------------
library(MASS)
set.seed(32608)
A <- matrix(rnorm(9),3,3)
ginv(A)
(A2 <- A[,1:2])
ginv(A2)
A2 %*% ginv(A2) %*% A2

#' 
#' Eigen value decomposition 
#' ===
#' 
#' - reference: <https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix>
#' 
#' - A (non-zero) vector $v$ of dimension $N$ is an eigenvector of a square $A\in \mathbb{R}^{N\times N}$ if it satisfies the linear equation
#' $$Av = \lambda v$$
#' 
#' - Let A be a square (N×N) matrix with N linearly independent eigenvectors, $q_i (i = 1, \ldots, N)$.
#' Then $A$ can be factorized as 
#' $$A = Q \Lambda Q^{-1}$$
#' 
#' ---
#' 
## ------------------------------------------------------------------------
A0 <- c(3,1,1,
       1,3,2,
       1,2,3)
A <- matrix(A0,3,3)
eigen(A)

Q <- eigen(A)$vectors
lambda <- eigen(A)$values

Q %*% t(Q)

Q %*% diag(lambda) %*% ginv(Q)

#' 
#' 
#' Singular value decomposition (SVD)
#' ===
#' 
#' - <https://en.wikipedia.org/wiki/Singular_value_decomposition>
#' 
#' - Singular value decomposition of $M \in \mathbb{R}^{m \times n}$ can be factorized as $U\Sigma V^\top$
#'     - $U \in \mathbb{R}^{m \times m}$ such that $UU^\top = I_m$.
#'     - $\Sigma \in \mathbb{R}^{m \times n}$ rectangular diagonal matrix.
#'     - $V \in \mathbb{R}^{n \times n}$ such that $VV^\top = I_n$.
#' 
#' - Equivalently, if $m \le n$, $M \in \mathbb{R}^{m \times n}$, $M = U\Sigma V^\top$
#'     - $U \in \mathbb{R}^{m \times m}$ such that $UU^\top = I_m$.
#'     - $\Sigma \in \mathbb{R}^{m \times m}$ rectangular diagonal matrix.
#'     - $V \in \mathbb{R}^{n \times m}$ such that $V^\top V= I_m$.
#' 
#' 
## ------------------------------------------------------------------------
M <- matrix(1:6,2,3) 
M
svd(M)

svdRes <- svd(M)
svdRes$u %*% diag(svdRes$d) %*% t(svdRes$v)

#' 
#' 
#' 
#' Determinant of a Matrix
#' ===
#' - det()
## ------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
det(S)

#' 
#' - rank (rank function has been implemented for the purpose of rank of a vector)
#' - To get the rank of a matrix:
#'     - Perform QR decomposition first.
#'     
## ------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
QR <- qr(S)
QR$rank

#' 
#' Column and Row Means
#' ===
#' - row mean
## ------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
apply(S,1,mean)
rowMeans(S)

#' 
#' - column mean
## ------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
apply(S,2,mean)
colMeans(S)

#' 
#' Generate code
#' ===
## ------------------------------------------------------------------------
knitr::purl("matrix.rmd", output = "matrix.R ", documentation = 2)

