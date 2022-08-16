#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday October 11, 2021"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Dimension reduction
#' ---
#' 
#' Outlines
#' ===
#' 
#' Several popular dimension reduction approaches
#' 
#' - Linear methods
#'   - Principal component analysis (PCA)
#'   - Singular value decomposition (SVD)
#'   - Non-negative matrix factorization (NMF)
#' - Non-linear methods
#'   - Multi dimension scaling (MDS)
#'   - T-Distributed Stochastic Neighbor Embedding (tSNE)
#' 
#' 
#' Principal component analysis
#' ===
#' 
#' - Can effectively perform dimension reduction
#' - A toy example of projecting 2-dimensional data onto 1-dimensional data.
#' 
#' ![](../figure/PCAexample.png)
#' 
#' Which direction to project?
#' ===
#' 
#' ![](../figure/PC_simu1.png)
#' 
#' - The information regarding the original data should be retained as much as possible.
#' - Or equivalently, the variance along the projected direction is maximized.
#' 
#' Projection results
#' ===
#' 
#' <img width="45%" src="../figure/PC_simu2.png"/>
#' <img width="45%" src="../figure/PC_simu3.png"/>
#' 
#' 
#' An example
#' ===
#' 
#' - Usually use the first principle component and the second principle component direction to visualize the data
#' - Motivating example: iris data, 150 observations, 4 variables (features)
## -------------------------------------------------------------------------------------------------
head(iris)
dim(iris)

#' 
#' ---
#' 
#' - Perform PCA
#'     - project these 150 observations from 4 dimensional space onto 2 dimensional space 
#'     - Usually, we need to standardize each feature to mean 0 and sd 1.
#' - visualize the data.
## -------------------------------------------------------------------------------------------------
iris.data <- iris[,1:4]
ir.pca <- prcomp(iris.data,
                 center = TRUE,
                 scale = TRUE) 
PC1 <- ir.pca$x[,"PC1"]
PC2 <- ir.pca$x[,"PC2"]
variance <- ir.pca$sdev^2 / sum(ir.pca$sdev^2)
v1 <- paste0("variance: ",signif(variance[1] * 100,3), "%")
v2 <- paste0("variance: ",signif(variance[2] * 100,3), "%")
plot(PC1, PC2, col=as.numeric(iris$Species),pch=19, xlab=v1, ylab=v2)
legend("topright", legend = levels(iris$Species), col =  unique(iris$Species), pch = 19)

#' 
#' Biplot
#' ===
#' 
#' - Data points: projection of the 150 observations onto the first two PCs in a Biplot.
#' - Project the observations onto arrow gives original (standardized) value of that variable.
#' - Arrow length: Variance of a variable.
#' - Angle between Arrows: Correlation = cos(angle).
#' 
#' 
## -------------------------------------------------------------------------------------------------
biplot(ir.pca)

#' 
#' ---
#' 
#' - Perform PCA (using ggplot)
#'     - project these 150 observations from 4 dimentional space onto 2 dimensional space 
#' - visualize the data.
## -------------------------------------------------------------------------------------------------
suppressWarnings(suppressMessages(library(tidyverse)))
iris.data <- iris[,1:4]
ir.pca <- prcomp(iris.data,
                 center = TRUE,
                 scale = TRUE) 
PC1 <- ir.pca$x[,"PC1"]
PC2 <- ir.pca$x[,"PC2"]
data_PCA <- tibble(PC1=PC1, PC2=PC2, Species = iris$Species)
variance <- ir.pca$sdev^2 / sum(ir.pca$sdev^2)
v1 <- paste0("variance: ",signif(variance[1] * 100,3), "%")
v2 <- paste0("variance: ",signif(variance[2] * 100,3), "%")
black.bold.text <- element_text(face = "bold", color = "black", size=20)

data_PCA %>% 
  ggplot() +
  aes(x=PC1,y=PC2,color=Species) + 
  geom_point() + 
  labs(x = v1, y=v2) +    
  theme_bw() + 
  theme(text = black.bold.text) 


#' 
#' 
#' Geometrical motivation for PCA (population version)
#' ===
#' 
#' - ${\bf x} \in \mathbb{R}^{G}$ is one sample with $G$ features.
#' - Without loss of generosity, 
#'     - assume $\mathbb{E}({\bf x}) = 0$
#'     - $Var({\bf x}) = \mathbb{E}({\bf x}^\top {\bf x}) = \Sigma \in \mathbb{R}^{G\times G}$
#' - We want to a direction (in unit length) $\alpha \in \mathbb{R}^{G}$ such that the variance of the projected value ($\alpha^\top {\bf x}$) is maximized.
#' $$\max_\alpha Var(\alpha^\top {\bf x}) = \max_\alpha \alpha^\top \Sigma \alpha, s.t. \|\alpha\|_2 = 1$$
#' - Claim the solution: $\alpha = {\bf v}_1$ which is the first eigen-value of $\Sigma$
#' 
#' 
#' Proof (page 1):
#' ===
#' 
#' - Since $\Sigma \in \mathbb{R}^{G\times G}$ is a symmetric and positive - definite matrix,
#' there exists matrix $V \in \mathbb{R}^{G\times G}$ and diagonal matrix $D \in \mathbb{R}^{G\times G}$ such that $$\Sigma = V D V^\top,$$
#' 
#' where $D = \begin{pmatrix}
#' d_1 & 0 & 0 & \\ 
#' 0  & d_2 &  &  \\ 
#' 0 &  & \ddots  & 0\\ 
#'  &  &  0 & d_G
#' \end{pmatrix},$
#' 
#' $d_1 \ge d_2 \ge \ldots \ge d_G \ge 0$ are eigenvalues.
#' ${\bf V} = ({\bf v}_1, {\bf v}_2, \ldots, {\bf v}_G)$ are eigen-vectors.
#' 
#' - ${\bf v}$'s are orthonormal 
#'     - ${\bf v}_i^\top {\bf v}_j = 0$ if $i \ne j$
#'     - $\|{\bf v}_g\|_2^2 = {\bf v}_g^\top {\bf v}_g = 1$, $\forall 1 \le g \le G$
#' 
#' - $\Sigma {\bf v}_g = d_g {\bf v}_g$ 
#' 
#' 
#' Proof (page 2):
#' ===
#' 
#' For any vector $\alpha \in \mathbb{R}^{G}$, it can be spanned as linear combination of ${\bf v}_i$'s
#' 
#' - $\alpha = \sum_{g=1}^G a_g {\bf v}_g$.
#'     - $\sum_{g=1}^G a_g^2 = 1$ if $\|\alpha\|_2^2 = 1$ 
#' 
#' - When $\|\alpha\|_2^2 = 1$,
#' 
#' $\begin{aligned}
#' Var(\alpha^\top {\bf x})  &= \alpha^\top \Sigma \alpha \\
#'                     &= (\sum_{g=1}^G a_g {\bf v}_g)^\top \Sigma (\sum_{g=1}^G a_g {\bf v}_g) \\
#'                     &= (\sum_{g=1}^G a_g {\bf v}_g)^\top(\sum_{g=1}^G a_g d_g {\bf v}_g) \\
#'                     &= \sum_{g=1}^G d_g a_g^2 \|{\bf v}_g\|_2^2 = \sum_{g=1}^G d_g a_g^2 \\
#'                     &\le \sum_{g=1}^G d_1 a_g^2 \\
#'                     &\le d_1 \\
#' \end{aligned}$
#' 
#' 
#' Proof (page 3):
#' ===
#' 
#' - If $\alpha = {\bf v}_1$, 
#' 
#' $\begin{aligned}
#' Var(\alpha^\top {\bf x})  &= \alpha^\top \Sigma \alpha \\
#'                           &= {\bf v}_1^\top \Sigma {\bf v}_1 \\
#'                           &=  d_1 \\
#' \end{aligned}$
#' 
#' - Therefore,  $\alpha = {\bf v}_1$ maximize $\max_\alpha Var(\alpha^\top {\bf x})$.
#' 
#' - And the projected random variable $L_1 = {\bf v}_1^\top {\bf x}$ has the largest variance $d_1$
#' 
#' - ${\bf v}_1 = (v_{11}, \ldots, v_{1G})$ are called loadings of PC1.
#' 
#' - Similarly, $L_2 = {\bf v}_2^\top {\bf x}$ has the largest variance among all possible projections 
#' orthogonal to $L_1$.
#' 
#' - Similarly, $L_g = {\bf v}_g^\top {\bf x}$ has the largest variance among all possible projections 
#' orthogonal to $L_1, L_2, \ldots, L_{g-1}$ for $1 \le g \le G$.
#' 
#' PCA 2D example
#' ===
#' 
#' ![](../figure/PCAfirstDirection.png)
#' 
#' 
#' Variance explained
#' ===
#' 
#' - $trace(\Sigma)$ is sum of variance of all features, is invariant of change of basis.
#'   - $trace(A) = trace(V\Sigma V^\top) = trace(\Sigma V^\top V) = trace(\Sigma)$
#'   - $trace(\Sigma) = \sum_g d_g = \sum_{g} Var(L_g)$.
#' - Define $r_g = \frac{d_g}{\sum_{g'} d_{g'}}$ where $r_g$ represents the proportion of variance explained by the $g^{th}$ PC.
#' - $R_g = \sum_{g'=1}^g r_{g'}$ is the cummulative proportion of variance explained by the first $g$ PCs.
#' 
#' 
#' PCA summary
#' ===
#' 
#' - Project the original data into orthonormal space ($\bf v_1, v_2, \ldots$) where $\bf v_g$ is the $g^{th}$ eigen vector of the covariance matrix $\Sigma$, such that the variance of the projected value along $\bf v_1$ is maximized, then along $\bf v_2$ is maximized, ...
#' - $L_1 = {\bf v_1}^\top {\bf x}$ is the projected value of $x$ on the first principal component
#'   - Also referred as the score of the first principal component
#'   - The new score is the linear combination of original data
#' - $L_2 = {\bf v_2}^\top {\bf x}$ is the projected value of $x$ on the second principal component
#'   - Also referred as the score of the second principal component
#' - These projection directions are the eigenvectors by applying eigenvalue decomposition to the covariance matrix $\Sigma$.
#' - Eigen value $d_1$ is the variance explained by $L_1$
#' - The proportion of variance explained by the first $g$ PCs is $r_g = \frac{d_g}{\sum_{g'} d_{g'}}$.
#' 
#' 
#' Example 1, HAPMAP data
#' ===
#' 
#' - The International HapMap Project was an organization that aimed to develop a haplotype map (HapMap) of the human genome, to describe the common patterns of human genetic variation. 
#' - A subset of HAPMAP data (n = 210):
#'     - CEU Utah residents with Northern and Western European ancestry from the CEPH collection
#'     - CHB Han Chinese in Beijing, China
#'     - JPT Japanese in Tokyo, Japan
#'     - YRI Yoruba in Ibadan, Nigeria
#' - 712,940 SNPs:
#'     - AA: 0
#'     - AB: 0.5
#'     - BB: 1
#' - Project 210 samples on 712,940 dimentional space onto a 2 dimentional space.
#'   - $210 \times 712,940$ matrix becomes $210 \times 2$.
#' 
#' HAPMAP data visualization
#' ===
#' 
#' ![](../figure/HAPMAP.png)
#' 
#' 
#' In HW (Visualize HAPMAP data using chromosome 1)
#' ===
#' 
#' - Human being have 23 pairs of chromosomes.
#' 
#' - Chromosome 1 of HAPMAP data is avaliable on /blue/phc6068/share/data/HAPMAP/chr1Data.Rdata
#' 
#' - Annotation is on /blue/phc6068/share/data/HAPMAP/relationships_w_pops_121708.txt
#' 
#' - These data are avaliable on hpg.rc.ufl.edu
#' 
#' - You can either perform the analysis on HiperGator, or downloaded the data to your local computer and perform the PCA analysis.
#' 
#' 
#' PCA steps (sample version)
#' ===
#' 
#' - **Center the data to mean 0 for each feature**. 
#' - Also standardize the varaince of each feature to be 1 after centering, otherwise feature with large variance will dominate.
#' - Calculate the sample covariance matrix.
#' - Perform eigen-value decomposition and obtain eigen-values and eigen-vectors.
#' - Project the original data onto the first eigen-vector, 
#'     - the resulting projected value on the first eigen-vector is called first principal component.
#'     - the first eigen-value is the total variance explained by the first principal component.
#' - Repeat the projection step for the $k^{th}$ eigen-value and eigen-vectors, $(1 \le k \le K)$
#'     - the direction of $k^{th}$ eigen-vector is perpendicular to all previous eigen-vectors.
#'     
#'     
#' How many principal component to use?
#' ===
#' - Select $K$ by eyeballing the inflection point of the scree plot, or total variance is greater than certain threshold ($>90\%$)
#' - Scree plot
#' 
#' ![](../figure/scree.png)
#' 
#' 
#' 
#' Example 2, yeast cycle data
#' ===
#' 
#' - The yeast cell cycle analysis project's goal is to identify all genes whose mRNA levels are regulated by the cell cycle. <http://genome-www.stanford.edu/cellcycle/>
#' 
#' ![https://le.ac.uk/-/media/uol/images/facilities/vgec/topics/cell-cycle-sc1.gif](https://le.ac.uk/-/media/uol/images/facilities/vgec/topics/cell-cycle-sc1.gif)
#' 
#' 
#' 
#' Yeast cell cycle data
#' ===
#' 
#' - Contains the expression profile of 800 genes $\times$ 77 samples 
#' - 800 genes can be categorized as 5 types by their peak time.
#'     1. G1
#'     2. S
#'     3. S/G2
#'     4. G2/M
#'     5. M/G1
#' - 77 yeast cells were first synchronized to the same G0 stage by four different chemicals: 
#'     - alpha
#'     - cdc15
#'     - cdc28
#'     - elu
#' 
#' 
#' Cell cycle data (will be on HW)
#' ===
#' 
## ---- eval=FALSE----------------------------------------------------------------------------------
## library(impute)
## raw = read.table("http://Caleb-Huo.github.io/teaching/data/cellCycle/cellCycle.txt",header=TRUE,as.is=TRUE)
## cellCycle = raw
## cellCycle[,2:78]<- impute.knn(as.matrix(raw[,2:78]))$data ## missing value imputation
## dim(cellCycle)

#' 
#' 
#' Yeast cell cycle data PCA for genes
#' ===
#' 
#' - Project 800 genes onto the first two PCs (samples are features).
#' - Each dot is a gene
#' 
#' ![](https://caleb-huo.github.io/teaching/data/cellCycle/cellCycleByGenes.png)
#' 
#' Yeast cycle data PCA for samples
#' ===
#' 
#' - Project all samples onto the first two PCs (genes are features).
#' - Each dot is a sample
#' 
#' ![](https://caleb-huo.github.io/teaching/data/cellCycle/cellCycleBySamples.png)
#' 
#' 
#' 
#' 
#' Singular value decomposition (SVD)
#' ===
#' 
#' 
#' - Singular value decomposition of $M \in \mathbb{R}^{m \times n}$ can be factorized as $U E V^\top$
#'   - If $m > n$
#'     - $U \in \mathbb{R}^{m \times n}$ such that $U^\top U = I_n$.
#'     - $E \in \mathbb{R}^{n \times n}$ diagonal matrix.
#'     - $V \in \mathbb{R}^{n \times n}$ such that $V^\top V = I_n$.
#' 
#'   - If $m \le n$, $M \in \mathbb{R}^{m \times n}$, $M = UE V^\top$
#'     - $U \in \mathbb{R}^{m \times m}$ such that $U^\top U= I_m$.
#'     - $E \in \mathbb{R}^{m \times m}$ rectangular diagonal matrix.
#'     - $V \in \mathbb{R}^{n \times m}$ such that $V^\top V= I_m$.
#' 
#' Equivalence between PCA and SVD
#' ===
#' 
#' - PCA
#'   - Covariance matrix $\Sigma = Var(x)$
#' $$\Sigma = V D V^\top, or \mbox{ } V^\top \Sigma V= D$$
#' 
#'   - $v_1$ is the first principal component
#'   - $d_1$ is the variance explained by $v_1$
#'   - $v_1^\top x$ is the score of x on the first principal component
#' 
#' - SVD
#'   - Suppose $X = U E  V^\top$, where $X$ has been standardized (mean 0 and sd 1).
#' 
#' $\begin{aligned}
#' V^\top X^\top X V &= V^\top (U E  V^\top)^\top U E  V^\top V \\
#'                   &= V^\top V E U^\top U E  V^\top V \\
#'                   &= E^2 
#' \end{aligned}$
#' 
#' 
#' ---
#' 
#' - Equivalence of PCA and SVD
#'   - $X_0 \in \mathbb{R}^{n \times p}$ is the observed data matrix.
#'   - $\hat{\Sigma} = \frac{1}{n - 1} X_0^\top X_0$
#' 
#' - Performing eigen-value decomposition on $X^\top X$ is equivalent to perform SVD on $X$
#'     - $V$ are eigen-vectors (projected directions)
#'     - $E^2$ are eigen-values (explained variances)
#'     - Projected value $XV = U E  V^\top V = U E$
#' 
#' 
#' 
#' 
#' Validate the SVD results using iris data
#' ===
#' 
## -------------------------------------------------------------------------------------------------
iris.data <- iris[,1:4]
iris.data.center <- scale(iris.data) ## mean 0 and std 1
asvd <- svd(iris.data.center)
UE <- asvd$u %*% diag(asvd$d)
PC1 <- UE[,1]
PC2 <- UE[,2]
variance <- asvd$d^2 / sum(asvd$d^2)
v1 <- paste0("variance: ",signif(variance[1] * 100,3), "%")
v2 <- paste0("variance: ",signif(variance[2] * 100,3), "%")
plot(PC1, PC2, col=as.numeric(iris$Species),pch=19, xlab=v1, ylab=v2)
legend("topright", legend = levels(iris$Species), col =  unique(iris$Species), pch = 19)

#' 
#' Important notes for PCA and SVD 
#' ===
#' - It is the best practice to scale each feature to
#'   - mean 0
#'   - sd 1
#' 
#' 
#' 
#' Non negetive matrix factorization (NMF)
#' ===
#' 
#' - Non-negative matrix factorization (NMF) is a dimension reduction algorithm where a non-negative matrix, X, is factorized into two non-negative matrices
#'     - W and H, all elements must be equal to or greater than zero.
#'     - The method has been applied in image recognition, text mining and bioinformatics.
#'     
#' $$\min_{W\in \mathbb{R}^{p\times r}, H \in \mathbb{R}^{r \times n}} \|X - WH\|_F,$$  
#' where $X \in \mathbb{R}^{p \times n}$, $\|X\|_F = \sqrt{\sum_{j=1}^p \sum_{i=1}^n X_{ji}^2}$
#' 
#' - W: score matrix
#' - H: loading matrix
#' 
#' 
#' NMF example
#' ===
#' 
## -------------------------------------------------------------------------------------------------
suppressMessages(library(NMF))

iris.data <- iris[,1:4]
anmf <- nmf(iris.data, rank = 2, method = "lee")

W <- anmf@fit@W
H <- anmf@fit@H

plot(W[,1], W[,2], col=as.numeric(iris$Species),pch=19)
legend("topright", legend = levels(iris$Species), col =  unique(iris$Species), pch = 19)

#' 
#' NMF example
#' ===
#' 
## -------------------------------------------------------------------------------------------------
basismap(anmf) ## W
coefmap(anmf) ## H

#' 
#' 
#' 
#' Multi dimensional Scaling (MDS)
#' ===
#' 
#' - Multi-dimensional scaling (MDS) aims to map data (distance or dissimilarity) structure to low dimensional space (usually 2D or 3D Euclidian space).
#'   - The input is dissimilarity matrix, not high dimentional data matrix (which is for PCA and SVD)
#' - Flight mileage of ten cities obtained from <http://www.webyer.com/travel/mileage_calculator/>.
#' 
## -------------------------------------------------------------------------------------------------
z <- read.delim("https://Caleb-Huo.github.io/teaching/data/UScityDistance/Mileage.txt", row.names=1)
knitr::kable(z, caption = "distance between cities (miles)")

#' 
#' Multi dimensional Scaling (MDS)
#' ===
#' 
#' - after apply MDS
#' 
#' ![](https://caleb-huo.github.io/teaching/data/UScityDistance/cityDistance.png)
#' 
#' 
#' - Will be on HW
#'   - Perform MDS on the following data, with both classical method and Sammon's stress.
#' 
#' 
#' Math behind MDS
#' ===
#' 
#' - objective function for classical MDS
#' $$ L = \min \sum_{i<j} (d_{ij} - \delta_{ij})^2$$
#' Assume the underlying data  $X \in \mathbb{n\times q}$ (Usually $q=2$) and $\|X_i - X_j\|_2 = d_{ij}$.
#' 
#' - parameters:
#'     - $i, j$: sample index
#'     - $\delta_{ij}$: the dissimilarity measure between object i and j in the original data space.
#'     - $d_{ij}$: the distance between the two objects after mapping to the targeted low-dimensional space.
#' 
#' - In the loss function above, large distances can dominate the optimization and ignore the local structure for pairs of short distances.
#'     - A possible modification may be to minimize the percent of squared loss.
#' $$ L = \min \sum_{i<j} (\frac{d_{ij} - \delta_{ij}}{\delta_{ij}})^2 $$
#' 
#' 
#' MDS with Sammon's stress
#' ===
#' 
#' - This modification, however, may over-emphasize the local structure and easily distort the global structure.
#' - A better balance between the two is the Sammon's stress.
#' $$ L = \min \sum_{i<j} (\frac{d_{ij} - \delta_{ij}}{\delta_{ij}})^2 \times w_{ij},$$
#' $w_{ij} = \delta_{ij} / (\sum_{i<j} \delta_{ij})$
#' 
#' Note:
#' 
#' - Reflection, translation and/or rotation (i.e. isometry) of an MDS solution is also an MDS solution since MDS only considers the preservation of the dissimilarity structure.
#' 
#' 
#' Classical mulditimensional scaling
#' ===
#' 
#' - $\delta_{ij}$ is the observed distance between sample $i$ and $j$, ($1\le i \le n$, $1\le j \le n$).
#' - $D = \{d_{ij}\}_{1\le i,j \le n}$ is derived from Euclidean distance of an unkown $n \times q$ data matrix $X \in \mathbb{R}^{n\times q}$ (Usually $q=2$).
#'     - $\|X_i - X_j\|_2 = d_{ij}$.
#'     - Such a solution is not unique, because if $X$ is the solution, $X^* = X + c$, $c \in \mathbb{R}^{q}$ is also solution, since $\|X_i^* - X_j^*\|_2 = \|(X_i + c) - (X_j + c)\|_2 = \|X_i - X_j\|_2$
#'     - So suppose $X_i$ is centered, (i.e. $\sum_{i=1}^n X_{iq} = 0$, for all $q$).
#'     - $B = XX^{\top}$, $B_{ij} = b_{ij}$
#' - The following relationship can be derived
#'     - $d_{ij}^2 = b_{ii} + b_{jj} - 2b_{ij}$
#'     - $\sum_{i=1}^n b_{ij} = 0$
#'     - $T = trace(B) = \sum_{i=1}^n b_{ii}$
#'     - $\sum_{i=1}^n d_{ij}^2 = T + nb_{jj}$
#'     - $\sum_{j=1}^n d_{ij}^2 = T + nb_{ii}$
#'     - $\sum_{i=1}^n \sum_{j=1}^n d_{ij}^2 = 2 n T$
#' 
#' 
#' Classical mulditimensional scaling continue
#' ===
#' 
#' - We can solve for:
#' $$b_{ij} = -1/2 (d_{ij}^2 - d{\cdot j}^2 - d{i\cdot}^2 + d_{\cdot\cdot}^2)$$
#'     - $d_{.j} = \frac{1}{n} \sum_{i = 1}^n d_{ij}$
#'     - $d_{i.} = \frac{1}{n} \sum_{j = 1}^n d_{ij}$
#'     - $d_{..} = \frac{1}{n^2} \sum_{i = 1}^n \sum_{j = 1}^n d_{ij}$
#' - Use $\delta_{ij}$ to approximate $d_{ij}$.
#' - A solution $X$ is then given by the eigen-decomposition of $B$. 
#'     - $B = V\Lambda V^{\top}$
#'     - $X = \Lambda^{1/2} V^{\top}$
#' 
#' 
#' MDS in R
#' ===
#' 
#' - classical: cmdscale()
#' - MDS with Sammon's stress: MASS::sammon()
#' 
#' Another MDS example - Genetic dissimilarity between races
#' ===
#' 
## -------------------------------------------------------------------------------------------------
z0 <- read.csv("https://Caleb-Huo.github.io/teaching/data/SNP_MDS/SNP_PCA.csv", row.names=1)
knitr::kable(z0, caption = "Genetic dissimilarity between races")

#' 
#' Another MDS example - Genetic dissimilarity between races
#' ===
#' 
## -------------------------------------------------------------------------------------------------
library(ggplot2)
library(ggrepel)

z <- as.dist(cbind(z0,NA))
z_mds = - cmdscale(z,k=2)

mdsDataFrame <- data.frame(Race=rownames(z_mds),x = z_mds[,1], y = z_mds[,2])

ggplot(data=mdsDataFrame, aes(x=x, y=y, label=Race)) +  
  geom_point(aes(color=Race)) + 
  geom_text_repel(data=mdsDataFrame, aes(label=Race)) +
  theme(legend.position = "none")


#' 
#' 
#' 
#' T-Distributed Stochastic Neighbor Embedding (tSNE)
#' ===
#' 
#' - tSNE is a non linear dimension reduction method
#' - well suited for visualizing high-dimensional data in low-dimensional space
#' - try to perserve the dissimilarity structure after dimention reduction
#' - very popular for single cell RNA-Seq data visualization
#' 
#' 
#' applications with tSNE
#' ===
#' 
## -------------------------------------------------------------------------------------------------
library(tsne)

data_tsne0 <- tsne(iris[,1:4])
data_tsne <- data.frame(tsne1 = data_tsne0[,1], tsne2 = data_tsne0[,2], species = iris$Species)

ggplot(data_tsne) +
aes(x = tsne1, y = tsne2, color = species) +
geom_point()

#' 
#' 
#' Math behind tSNE
#' ===
#' 
#' - High dimentional data (original data):
#'   - $x_i \in \mathbb{R}^G$
#'   - Distance between sample $i$ and $j$: $p_{ij} = D_x(x_i, x_j)$
#' - Low dimentional data (projected data):
#'   - $y_i \in \mathbb{R}^2$ (2 dimension, can be 1 or 3 or more)
#'   - Distance between sample $i$ and $j$: $q_{ij} =D_y(y_i, y_j)$
#' - Objective function of tSNE
#'   - $min_{y_i} \sum_{i\ne j} dist(p_{ij}, q_{ij})$
#'   - $p_{i,j}$
#'     - $p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2/2\sigma_i)}{\sum_{k \ne i}\exp(-\|x_i - x_k\|^2/2\sigma_i)}$
#'     - $p_{j|i}$ is standardized Gaussian density function.
#'     - $p_{i,j} = \frac{p_{j|i} + p_{i|j}}{2}$
#'   - $q_{ij}$
#'     - $q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{l \ne k}(1 + \|y_l - y_k\|^2)^{-1}}$
#'     - $q_{ij}$ is standardized t distribution with degree of freedom 1 (Cauchy distribution)
#'   - $dist$
#'     - $dist(p_{ij}, q_{ij}) = KL(P||Q) = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$
#'     - KL represents KL divergence, which measures the distance between probabilities. 
#' 
#' 
