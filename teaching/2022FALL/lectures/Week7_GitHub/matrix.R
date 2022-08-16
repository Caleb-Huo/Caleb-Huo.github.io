#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Wednesday Oct 06, 2021"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Matrix operation
#' ---
#' 
#' Outline
#' ===
#' 
#' - Algebra notation review
#' - Matrix operation in R
#' 
#' 
#' Algebra notation
#' ===
#' 
#' - scalar: $x \in \mathbb{R}$
#' - vector: $\textbf{x} \in \mathbb{R}^p$
#'   - by default, we write a vector as a column
#' $$
#' \textbf{x}=
#' \begin{pmatrix}
#' x_1 \\ 
#' x_2 \\ 
#' \ldots \\ 
#' x_p 
#' \end{pmatrix}
#' $$
#'   
#'   - or we can write as the transpose of a row vector
#' 
#' $$
#' \textbf{x} = (x_1, x_2, \ldots, x_p)^\top
#' $$  
#' 
#' - matrix: $\textbf{X} \in \mathbb{R}^{n\times p}$
#' 
#' $$
#' \textbf{X}=
#' \begin{pmatrix}
#' x_{11} & x_{12} & \ldots & x_{1p} \\ 
#' x_{21} & x_{22} & \ldots & x_{2p} \\ 
#' \ldots \\ 
#' x_{n1} & x_{n2} & \ldots & x_{np} 
#' \end{pmatrix}
#' $$
#' 
#' 
#' Cross product of two vectors
#' ===
#' 
#' - $\textbf{a} = (a_1, \ldots, a_p)^\top \in \mathbb{R^p}$
#' - $\textbf{b} = (b_1, \ldots, b_p)^\top \in \mathbb{R^p}$
#' 
#' $$
#' \textbf{a}^\top \textbf{b} = \textbf{b}^\top \textbf{a} 
#' = a_1b_1 + a_2b_2 + \ldots + a_p b_p = \sum_{i=1}^p a_ib_i
#' $$
#' 
#' - match dimension:
#'   - $\textbf{a}^\top \in \mathbb{R}^{1\times p}$
#'   - $\textbf{b} \in \mathbb{R}^{p\times 1}$
#'   - $\textbf{a}^\top \textbf{b} \in \mathbb{R}^{1\times 1} = \mathbb{R}$
#' 
#' - a special case
#' $$
#' \textbf{a}^\top \textbf{a}
#' = a_1^2 + a_2^2 + \ldots a_p^2
#' = \sum_{i=1}^p a_i^2
#' = \|\textbf{a}\|_2^2
#' $$
#' $$\|\textbf{a}\|_2 = \sqrt{\sum_{i=1}^p a_i^2}$$
#' 
#' Matrix notation
#' ===
#' 
#' - a vector $\textbf{a} = (a_1, \ldots, a_n)^\top \in \mathbb{R}^n$
#' - a matrix: $\textbf{X} \in \mathbb{R}^{n\times p}$
#'   - $\textbf{X} = (\textbf{X}_1, \textbf{X}_2, \ldots, \textbf{X}_p)$
#'   - Each $\textbf{X}_i \in \mathbb{R}^n$
#' 
#' $$
#' \textbf{a}^\top\textbf{X} = \textbf{a}^\top (\textbf{X}_1, \textbf{X}_2, \ldots, \textbf{X}_p)
#' = (\textbf{a}^\top \textbf{X}_1, \textbf{a}^\top\textbf{X}_2, \ldots, \textbf{a}^\top\textbf{X}_p)
#' \in \mathbb{R}^{1 \times p}
#' $$
#' 
#' Matrix notation
#' ===
#' 
#' - matrix $\textbf{X} \in \mathbb{R}^{n \times p}$: 
#' $$\textbf{X}=
#' \begin{pmatrix}
#' x_{11} & x_{12} & \ldots & x_{1p} \\ 
#' x_{21} & x_{22} & \ldots & x_{2p} \\ 
#' \ldots \\ 
#' x_{n1} & x_{n2} & \ldots & x_{np} 
#' \end{pmatrix}
#' $$
#' 
#' 
#' - matrix $\textbf{Y} \in \mathbb{R}^{p \times m}$: 
#' $$\textbf{Y}=
#' \begin{pmatrix}
#' y_{11} & y_{12} & \ldots & y_{1m} \\ 
#' y_{21} & y_{22} & \ldots & y_{2m} \\ 
#' \ldots \\ 
#' y_{p1} & y_{p2} & \ldots & y_{pm} 
#' \end{pmatrix}
#' $$
#' 
#' - matrix $\textbf{X} \textbf{Y} \in \mathbb{R}^{n \times m}$: 
#' 
#' $$\textbf{X} \textbf{Y} = 
#' \begin{pmatrix}
#' \sum_{i=1}^p x_{1i}y_{i1} & \sum_{i=1}^p x_{1i}y_{i2} & \ldots & \sum_{i=1}^p x_{1i}y_{im} \\ 
#' \sum_{i=1}^p x_{2i}y_{i1} & \sum_{i=1}^p x_{2i}y_{i2} & \ldots & \sum_{i=1}^p x_{2i}y_{im} \\ 
#' \ldots \\ 
#' \sum_{i=1}^p x_{ni}y_{i1} & \sum_{i=1}^p x_{ni}y_{i2} & \ldots & \sum_{i=1}^p x_{ni}y_{im} 
#' \end{pmatrix}
#' $$
#' 
#' 
#' 
#' Matrix notation
#' ===
#' 
#' - matrix $\textbf{X} \in \mathbb{R}^{n \times p}$: 
#'   - $\textbf{x}_{i} \in \mathbb{R}^p$, $i = 1,2, \ldots, n$
#'   
#' $$\textbf{X}=
#' \begin{pmatrix}
#' \textbf{x}_{1}^\top\\ 
#' \ldots \\ 
#' \textbf{x}_{n}^\top
#' \end{pmatrix}
#' $$
#' 
#' 
#' - matrix $\textbf{Y} \in \mathbb{R}^{p \times m}$: 
#'   - $\textbf{y}_{j} \in \mathbb{R}^p$, $j = 1,2, \ldots, m$
#' 
#' $$\textbf{Y}=
#' (\textbf{y}_{1}, \ldots, \textbf{y}_{m})
#' $$
#' 
#' - matrix $\textbf{X} \textbf{Y} \in \mathbb{R}^{n \times m}$: 
#' 
#' $$\textbf{X} \textbf{Y} = 
#' \begin{pmatrix}
#' \textbf{x}_{1}^\top \textbf{y}_{1} & \textbf{x}_{1}^\top \textbf{y}_{2} & \ldots & \textbf{x}_{1}^\top \textbf{y}_{m} \\ 
#' \textbf{x}_{2}^\top \textbf{y}_{1} & \textbf{x}_{2}^\top \textbf{y}_{2} & \ldots & \textbf{x}_{2}^\top \textbf{y}_{m} \\ 
#' \ldots \\ 
#' \textbf{x}_{n}^\top \textbf{y}_{1} & \textbf{x}_{n}^\top \textbf{y}_{2} & \ldots & \textbf{x}_{n}^\top \textbf{y}_{m} \\ 
#' \end{pmatrix}
#' $$
#' 
#' 
#' 
#' Inverse of a matrix
#' ===
#' 
#' - matrix $\textbf{Y} \in \mathbb{R}^{p \times p}$: 
#' - $I_{p\times p}$ is the identity matrix (the diagnals are 1 and 0 elsewhere)
#' 
#' $$\textbf{Y}^{-1} \textbf{Y} = I$$
#' $$\textbf{Y} \textbf{Y}^{-1} = I$$
#' 
#' - We <span style="color:red">couldn't</span> write in the following way, since division is not defined for matrix.
#' 
#' $$\textbf{Y}^{-1} = \frac{I}{\textbf{Y}}$$
#' 
#' 
#' 
#' Create a matrix
#' ===
#' - an example from ?matrix
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat <- matrix(c(1,2,3, 11,12,13), nrow = 2, ncol = 3, byrow = FALSE,
               dimnames = list(c("row1", "row2"),
                               c("C.1", "C.2", "C.3")))
mdat

#' 
#' - an alternative way
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat <- c(1,2,3, 11,12,13)
dim(mdat) <- c(2,3)
rownames(mdat) <- c("row1", "row2")
colnames(mdat) <- c("C.1", "C.2", "C.3")
mdat

#' 
#' Appending a column  or a row to a matrix
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
dim(mdat)

#' - appending a column
## ----------------------------------------------------------------------------------------------------------------------------------------------
mat2 <- cbind(mdat, c(3,4))
dim(mat2)

#' 
#' - appending a row
## ----------------------------------------------------------------------------------------------------------------------------------------------
mat3 <- rbind(mdat, c(2,5,8))
dim(mat3)

#' 
#' 
#' Subset of a matrix
#' ===
#' - by numeric index
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat[1,1]
mdat[1,1:2]
mdat[1,]
mdat[1,-1]
mdat[,]

#' 
#' ---
#' 
#' - by names
## ----------------------------------------------------------------------------------------------------------------------------------------------
dimnames(mdat); rownames(mdat); colnames(mdat)
mdat["row1", "C.1"]
mdat["row2", c("C.2", "C.3")]
mdat["row1",-2] ## mixed usage

#' 
#' Transponse of a Matrix
#' ===
#' - t()
## ----------------------------------------------------------------------------------------------------------------------------------------------
mat1 <- matrix(1:6,nrow=2,ncol=3)
mat1
t(mat1)

#' 
#' Element-wise Multiplication by a Scalar/Vector/Matrix
#' ===
#' - by a scalar
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat * 3

#' 
#' - by a **column** vector
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat * c(1,2)

#' 
#' 
#' - by a matrix of the same dimension
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat * matrix(1:6,nrow=2,ncol=3)

#' 
#' 
#' 
#' Addition/Subtraction by a Scalar/Vector/Matrix
#' ===
#' - by a scalar
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat + 3

#' 
#' - by a **column** vector
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat - c(1,2)

#' 
#' 
#' - by a matrix of the same dimension
## ----------------------------------------------------------------------------------------------------------------------------------------------
mdat + matrix(1:6,nrow=2,ncol=3)

#' 
#' 
#' 
#' Matrix Multiplication
#' ===
#' - %*%
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------------------------------------------------------
S <- matrix(1:9,3,3)
S
diag(S)
diag(c(1,4,7))
diag(3)

#' 
#' 
#' Inverse of a Matrix
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------------------------------------------------------
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
## ----------------------------------------------------------------------------------------------------------------------------------------------
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
#' - A (non-zero) vector $\textbf{v} \in \mathbb{R}^N$  is an eigenvector of a square matrix $A\in \mathbb{R}^{N\times N}$ if it satisfies the linear equation
#' $$A\textbf{v} = \lambda \textbf{v}$$
#' where $\lambda \in \mathbb{R}$ is the eigenvalue corresponding to $\textbf{v}$.
#' Since if $\textbf{v}$ is eigenvector, $b\textbf{v}$ ($b\in \mathbb{R}$) is also eigenvector, we restrict $\|\textbf{v}\|_2 = 1$
#' 
#' - Let $A \in \mathbb{R}^{N \times N}$ be a square  matrix with N linearly independent eigenvectors, $\textbf{v}_i (i = 1, \ldots, N)$.
#' Then $A$ can be factorized as 
#' $$A = V \Lambda V^{-1}$$
#'   - $V=(\textbf{v}_{1}, \ldots, \textbf{v}_{N})$ 
#'   - $\Lambda$ is a diagnal matrix 
#'   ```
#'   diag(lambda_1, ..., lambda_N)
#'   ```
#'   - $V$ is orthonormal in the sense
#' $$VV^\top = I$$
#'   
#' 
#'   
#' ---
#' 
#' - Eigen value decomposition
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
A0 <- c(3,1,1,
       1,3,2,
       1,2,3)
A <- matrix(A0,3,3)
eigen(A)


#' 
#' 
#' ---
#' 
#' - Verify
#'   - $\Lambda$ is a diagnal matrix
#'   - $V$ is orthonormal in the sense $VV^\top = I$, 
#'   - $A = V \Lambda V^{-1}$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
V <- eigen(A)$vectors
lambda <- eigen(A)$values

V %*% t(V)

V %*% diag(lambda) %*% ginv(V)

#' 
#' 
#' ---
#' 
#' - Verify $A\textbf{v} = \lambda \textbf{v}$
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
A %*% V[,1]
lambda[1] * V[,1]

A %*% V[,2]
lambda[2] * V[,2]

#' 
#' 
#' 
#' 
#' Singular value decomposition (SVD)
#' ===
#' 
#' 
#' - Singular value decomposition of $M \in \mathbb{R}^{m \times n}$ can be factorized as $U\Sigma V^\top$
#'   - If $m > n$
#'     - $U \in \mathbb{R}^{m \times n}$ such that $U^\top U = I_n$.
#'     - $\Sigma \in \mathbb{R}^{n \times n}$ diagonal matrix.
#'     - $V \in \mathbb{R}^{n \times n}$ such that $V^\top V = I_n$.
#' 
#'   - If $m \le n$, $M \in \mathbb{R}^{m \times n}$, $M = U\Sigma V^\top$
#'     - $U \in \mathbb{R}^{m \times m}$ such that $U^\top U= I_m$.
#'     - $\Sigma \in \mathbb{R}^{m \times m}$ rectangular diagonal matrix.
#'     - $V \in \mathbb{R}^{n \times m}$ such that $V^\top V= I_m$.
#' 
#' ---
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
M <- matrix(1:6,2,3) 
M
svd(M)

svdRes <- svd(M)
svdRes$u %*% diag(svdRes$d) %*% t(svdRes$v)

#' 
#' ---
#' 
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
t(svdRes$u) %*% svdRes$u

#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------
t(svdRes$v) %*% svdRes$v

#' 
#' Determinant and trace of a Matrix
#' ===
#' 
#' - $|S|$, where $S \in \mathbb{R}^{p \times p}$
#'   - det()
## ----------------------------------------------------------------------------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
det(S)

#' 
#' 
#' - $trace(S)$, where $S \in \mathbb{R}^{p \times p}$
#'   - $trace(S) = \sum_{i=1}^p S_{ii}$
#'   
## ----------------------------------------------------------------------------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
sum(diag(S))

#' 
#' 
#' Rank of a Matrix
#' ===
#' - Rank represents number of independent columns in a matrix
#' - rank (rank function has been implemented for the purpose of rank [order] of a vector)
#'   - To get the rank of a matrix, perform QR decomposition first.
#'     
## ----------------------------------------------------------------------------------------------------------------------------------------------
S <- matrix(c(4,4,-2,2,6,2,2,8,4),3,3)
QR <- qr(S)
QR$rank

