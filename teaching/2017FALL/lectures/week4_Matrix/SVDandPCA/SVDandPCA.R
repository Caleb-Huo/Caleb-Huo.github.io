#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday September 25, 2017"
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
#' - Principal component analysis (PCA)
#' - Singular value decomposition (SVD)
#' - Non-negative matrix factorization (NMF)
#' - Multi dimension scaling (MDS)
#' 
#' Principal component analysis
#' ===
#' 
#' - Can effectively perform dimension reduction
#' - A toy example of projecting 2-dimensional data onto 1-dimensional data.
#' ![](../figure/PCAexample.png)
#' 
#' Which direction to project?
#' ===
#' 
#' ![](../figure/PC_simu.png)
#' 
#' - The variance along the projected direction is maximized.
#' 
#' Another motivating example
#' ===
#' 
#' - Usually use first principle component and second principle component direction to visualize the data
#' - Motivating example: iris data, 150 observations, 4 variables (features)
## ------------------------------------------------------------------------
head(iris)
dim(iris)

#' 
#' ---
#' 
#' - Perform PCA
#'     - project these 4 features onto 2 dimensional space 
#' - visualize the data.
## ------------------------------------------------------------------------
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
#' - Data points: Projection on first two PCs Distance in Biplot.
#' - Projection of sample onto arrow gives original (scaled) value of that variable.
#' - Arrowlength: Variance of variable.
#' - Angle between Arrows: Correlation.
#' 
#' 
## ------------------------------------------------------------------------
biplot(ir.pca)

#' 
#' 
#' Geometrical motivation for PCA (population version)
#' ===
#' 
#' - ${\bf x} \in \mathbb{R}^{G}$ is one sample with $G$ features.
#' - Without loss of generosity, 
#'     - assume $\mathbb{E}({\bf x}) = 0$
#'     - $Var({\bf x}) = \mathbb{E}({\bf x}^\top {\bf x}) = \Sigma \in \mathbb{R}^{G\times G}$
#' - We want to a direction $\alpha \in \mathbb{R}^{G}$ such that the variance of the projected value ($\alpha^\top {\bf x}$) is maximized.
#' $$\max_\alpha Var(\alpha^\top {\bf x}) = \max_\alpha \alpha^\top \Sigma \alpha, s.t. \|\alpha\| = 1$$
#' - solution: $\alpha = {\bf v}_1$ which is the first eigen-value of $\Sigma$
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
#' - ${\bf V}$ are orthonormal 
#'     - ${\bf v}_i^\top {\bf v}_j = 0$ if $i \ne j$
#'     - $\|{\bf v}_g\|^2 = {\bf v}_g^\top {\bf v}_g = 1$, $\forall 1 \le g \le G$
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
#'     - $\sum_{g=1}^G a_g^2 = 1$ if $\|\alpha\|^2 = 1$ 
#' 
#' - When $\|\alpha\| = 1$,
#' 
#' $\begin{aligned}
#' Var(\alpha^\top {\bf x})  &= \alpha^\top \Sigma \alpha \\
#'                     &= (\sum_{g=1}^G a_g {\bf v}_g)^\top \Sigma (\sum_{g=1}^G a_g {\bf v}_g) \\
#'                     &= (\sum_{g=1}^G a_g {\bf v}_g)^\top(\sum_{g=1}^G a_g d_g {\bf v}_g) \\
#'                     &= \sum_{g=1}^G d_g a_g^2 \|{\bf v}_g\|^2 = \sum_{g=1}^G d_g a_g^2 \\
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
#' Similarly, $L_g = {\bf v}_g^\top {\bf x}$ has the largest variance among all possible projections 
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
#' - $trace(\Sigma)$ is sum of variance of all covariates, is invariant of change of basis.
#' - $trace(\Sigma) = \sum_{g} Var(L_g) = \sum_g d_g$.
#' - Define $r_g = \frac{d_g}{\sum_{g'} d_{g'}}$ where $r_g$ represents the proportion of variance explained by the $g^{th}$ PC.
#' - $R_g = \sum_{g'=1}^g r_{g'}$ is the cummulative proportion of variance explained by the first $g$ PCs.
#' 
#' 
#' PCA summary
#' ===
#' 
#' - Project the original data into orthonormal space ($\bf v_1, v_2, \ldots$) such that the variance of the projected value along $\bf v_1$ is maximized, then along $\bf v_2$ is maximized, ...
#' - $L_1 = {\bf v_1}^\top {\bf x}$ is called the first principal component, $L_2 = {\bf v_2}^\top {\bf x}$ is called the second principal component...
#' - These projection directions are the eigenvectors by applying eigenvalue decomposition to the covariance matrix.
#' - The proportion of variance explained by the first $g$ PCs is $r_g = \frac{d_g}{\sum_{g'} d_{g'}}$.
#' 
#' 
#' Example 1, HAPMAP data
#' ===
#' 
#' - The International HapMap Project was an organization that aimed to develop a haplotype map (HapMap) of the human genome, to describe the common patterns of human genetic variation. 
#' - A subset of HAPMAP data:
#'     - CEU Utah residents with Northern and Western European ancestry from the CEPH collection
#'     - CHB Han Chinese in Beijing, China
#'     - JPT Japanese in Tokyo, Japan
#'     - YRI Yoruba in Ibadan, Nigeria
#' - 712,940 SNPs:
#'     - AA: 0
#'     - AB: 0.5
#'     - BB: 1
#' 
#' 
#' HAPMAP data visualization
#' ===
#' 
#' ![](../figure/HAPMAP.png)
#' 
#' 
#' In class exercise (Visualize HAPMAP data using chromosome 1)
#' ===
#' 
#' - Human being have 23 chromosomes.
#' 
#' - Chromosome 1 of HAPMAP data is avaliable on /ufrc/phc6068/share/data/HAPMAP/chr1Data.Rdata
#' 
#' - Annotation is on /ufrc/phc6068/share/data/HAPMAP/relationships_w_pops_121708.txt
#' 
#' - These data are avaliable on hpg2.rc.ufl.edu
#' 
#' - You can either perform the analysis on HiperGator, or downloaded the data to your local computer and perform the PCA analysis.
#' 
#' 
#' PCA steps (sample version)
#' ===
#' 
#' - Center the data to mean 0 for each feature. (Sometimes also standardized the varaince to be 1).
#' - Calculate the sample covariance matrix.
#' - Perform eigen-value decomposition and obtain eigen-values and eigen-vectors.
#' - Project the original data onto the first eigen-vector, 
#'     - the resulting projected value on the first eigen-vector is called first principal component.
#'     - the first eigen-value is the total variance explained by the first principal component.
#' - Repeat the projection step for the $k^{th}$ eigen-value and eigen-vectors, $(1 \le k \le K)$
#'     - the direction of $k^{th}$ eigen-vector is perpendicular to all previous eigen-vectors.
#' - Select $K$ using scree plot, or total variance is greater than certain threshold ($>90\%$)
#' 
#' How many principal component to use?
#' ===
#' 
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
#' - contains the expression profile of 800 genes and their annotated cell cycle stage
#'     1. G1
#'     2. S
#'     3. S/G2
#'     4. G2/M
#'     5. M/G1
#' - Yeast cells were first synchronized to the same G0 stage by four different chemicals: 
#'     - alpha
#'     - cdc15
#'     - cdc28
#'     - elu
#' 
#' Example 2, yeast cycle data PCA for genes
#' ===
#' 
#' ![](https://caleb-huo.github.io/teaching/data/cellCycle/cellCycleByGenes.png)
#' 
#' Example 2, yeast cycle data PCA for samples
#' ===
#' 
#' ![](https://caleb-huo.github.io/teaching/data/cellCycle/cellCycleBySamples.png)
#' 
#' Will be on HW
#' ===
#' 
#' Repeat these two PCA plot
#' 
## ---- eval=FALSE---------------------------------------------------------
## library(impute)
## raw = read.table(url("http://Caleb-Huo.github.io/teaching/data/cellCycle/cellCycle.txt"),header=TRUE,as.is=TRUE)
## cellCycle = raw
## cellCycle[,2:78]<- impute.knn(as.matrix(raw[,2:78]))$data ## missing value imputation

#' 
#' Singular value decomposition (SVD)
#' ===
#' 
#' - $X \in \mathbb{R}^{n \times p}$ is our data matrix with $n$ samples and $p$ features.
#'     - Without loss of generosity, $n < p$
#' - $X$ can be factorized as $U E_0 V_0^\top$
#'     - $U \in \mathbb{R}^{n \times n}$ such that $UU^\top = I_n$.
#'     - $E_0 \in \mathbb{R}^{n \times p}$. With $E_0 = (E, {\bf 0}_{(p-n) \times n})$, $E\in \mathbb{R}^{n \times n}$ is a diagnol matrix.
#'     - $V_0 \in \mathbb{R}^{p \times p}$. With $V_0 = (V, {\bf 0}_{p \times (p-n)})$, $V\in \mathbb{R}^{p \times n}$ such that $V^\top V= I_n$.
#' 
#' - Equivalently, $X = U E  V^\top$
#'     - where $E = \begin{pmatrix}
#' e_1 & 0 & 0 & \\ 
#' 0  & e_2 &  &  \\ 
#' 0 &  & \ddots  & 0\\ 
#'  &  &  0 & e_n
#' \end{pmatrix},$
#' 
#' 
#' Equivalence between PCA and SVD
#' ===
#' 
#' $\begin{aligned}
#' V^\top X^\top X V &= V^\top (U E  V^\top)^\top U E  V^\top V \\
#'                   &= V^\top V E U^\top U E  V^\top V \\
#'                   &= E^2 
#' \end{aligned}$
#' 
#' - Performing eigen-value decomposition on $X^\top X$ is equivalent to perform SVD on $X$
#'     - $V$ are eigen-vectors (projected directions)
#'     - $E^2$ are eigen-values (explained variances)
#'     - Projected value $XV = U E  V^\top V = U E$
#' 
#' 
#' Validate the SVD results using iris data
#' ===
#' 
## ------------------------------------------------------------------------
iris.data <- iris[,1:4]
iris.data.center <- scale(iris.data)
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
#' 
#' Multi dimensional Scaling (MDS)
#' ===
#' 
#' - Multi-dimensional scaling (MDS) aims to map data (distance or dissimilarity) structure to low dimensional space (usually 2D or 3D Euclidian space).
#' 
#' - Flight mileage of ten cities obtained from <http://www.webyer.com/travel/mileage_calculator/>.
#' 
## ------------------------------------------------------------------------
z <- read.delim(url("https://Caleb-Huo.github.io/teaching/data/UScityDistance/Mileage.txt"), row.names=1)
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
#' Will be on HW
#' ===
#' 
#' Perform MDS on the following data, with both classical method and Sammon's stress.
#' 
#' 
#' Math behind MDS
#' ===
#' 
#' - objective function for classical MDS
#' $$ L = \min \sum_{i<j} (d_{ij} - \delta_{ij})^2$$
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
#' $$ L = \min (\frac{d_{ij} - \delta_{ij}}{\delta_{ij}})^2 \times w_{ij},$$
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
#' - $D = \{d_{ij}\}_{1\le i,j \le n}$ is derived from Euclidean distance of an unnkown $n \times q$ data matrix $X \in \mathbb{n\times q}$ (Usually $q=2$).
#'     - $\|X_i - X_j\| = d_{ij}$.
#'     - Such a solution is not unique, because if $X$ is the solution, $X^* = X + c$, $c \in \mathbb{q}$ is also solution, since $\|X_i^* - X_j^*\| = \|(X_i + c) - (X_j + c)\| = \|X_i - X_j\|$
#'     - So suppose $X_i$ is centered, (i.e. $\sum_{i=1}^n X_{iq} = 0$, for all $q$).
#'     - $B = XX^\top$, $B_{ij} = b_{ij}$
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
#'     - $B = V\Lambda V^\top$
#'     - $X = \Lambda^{1/2} V^\top$
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
## ------------------------------------------------------------------------
z0 <- read.csv(url("https://Caleb-Huo.github.io/teaching/data/SNP_MDS/SNP_PCA.csv"), row.names=1)
knitr::kable(z0, caption = "Genetic dissimilarity between races")

#' 
#' Another MDS example - Genetic dissimilarity between races
#' ===
#' 
## ------------------------------------------------------------------------
library(ggplot2)
library(ggrepel)

z <- as.dist(cbind(z0,NA))
z_mds = - cmdscale(dist(z),k=2)

mdsDataFrame <- data.frame(Race=rownames(z_mds),x = z_mds[,1], y = z_mds[,2])

ggplot(data=mdsDataFrame, aes(x=x, y=y, label=Race)) + geom_point(aes(color=Race)) + geom_text_repel(data=mdsDataFrame, aes(label=Race))


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
#' 
#' 
#' 
#' NMF example
#' ===
#' 
## ------------------------------------------------------------------------
library(NMF) 

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
## ------------------------------------------------------------------------
basismap(anmf) ## W
coefmap(anmf) ## H

#' 
#' ---
#' 
## ------------------------------------------------------------------------
knitr::purl("SVDandPCA.Rmd", output = "SVDandPCA.R ", documentation = 2)

