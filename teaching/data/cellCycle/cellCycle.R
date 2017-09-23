setwd("/Users/zhuo/Dropbox/Caleb-Huo.github.io/teaching/data/cellCycle")
raw = read.table(file="cellCycle.txt",header=TRUE,as.is=TRUE)

library(impute)
library(ggplot2)

cellCycle = raw
cellCycle[,2:78]<- impute.knn(as.matrix(raw[,2:78]))$data

aData <- cellCycle[,2:78]
atype <- cellCycle[,79]

PCA_cellCycle_gene <- prcomp(aData,
                 center = TRUE,
                 scale. = TRUE) 
#
PCValues <- data.frame(PCA_cellCycle_gene$x, type=as.factor(atype))
varianceExplained <- PCA_cellCycle$sdev^2/sum(PCA_cellCycle$sdev^2)
varianceExplained_PC1 <- paste0("variance explained: ",signif(varianceExplained[1]* 100,3),"%")
varianceExplained_PC2 <- paste0("variance explained: ",signif(varianceExplained[2]* 100,3),"%")

png("cellCycleByGenes.png")
ggplot(PCValues, aes(x=PC1, y=PC2, color=type)) + geom_point() + labs(x = varianceExplained_PC1, y = varianceExplained_PC2) +
theme(text = element_text(size=20)) 
dev.off()


PCA_cellCycle_sample <- prcomp(t(aData),
                 center = TRUE,
                 scale. = TRUE) 
#
phase <- gsub("[0-9]|[.]","",colnames(aData))
phase[grep("cdc", colnames(aData))] <- sapply(colnames(aData),function(x) strsplit(x,"[.]")[[1]][1])[grep("cdc", colnames(aData))] 

PCValues <- data.frame(PCA_cellCycle_sample$x, phase=as.factor(phase))
varianceExplained <- PCA_cellCycle_sample$sdev^2/sum(PCA_cellCycle_sample$sdev^2)
varianceExplained_PC1 <- paste0("variance explained: ",signif(varianceExplained[1]* 100,3),"%")
varianceExplained_PC2 <- paste0("variance explained: ",signif(varianceExplained[2]* 100,3),"%")

subsetPhase <- PCValues[- grep("cl", PCValues$phase), ]

png("cellCycleBySamples.png")
ggplot(subsetPhase, aes(x=PC1, y=PC2, color=phase)) + geom_path() + 
facet_wrap(~phase)+ labs(x = varianceExplained_PC1, y = varianceExplained_PC2) +
theme(text = element_text(size=20)) 
dev.off()



# hierarchical clustering --------------------------------------------------

cellCycle.hclust<-hclust(dist(cellCycle[, 2:78]) , method="ward")
plot(cellCycle.hclust)
clusterHclust <- cutree(cellCycle.hclust, k = 5)
table(clusterHclust,cellCycle[,79])

# k-means -----------------------------------------------------------------
set.seed(15213)
cellCycle.kmeans<-kmeans(as.matrix(cellCycle[,2:78]), 5,nstart=20)
table(cellCycle.kmeans$cluster,cellCycle[,79])



#### ARI


# a) ----------------------------------------------------------------------

library(mclust)
mycombn <- function(n){
  if(n<2) return(0)
  return(n*(n-1)/2)
}


myARI <- function(a,b){
  if(length(a)!=length(b)) return("different vector length")
  tab = table(a,b)
  ni = apply(tab,1,sum)
  nj = apply(tab,2,sum)
  nn = sum(ni)
  
  RI = sum(sapply(tab,mycombn))
  maxa = sum(sapply(ni,mycombn))
  maxb = sum(sapply(nj,mycombn))
  expectNull = maxa*maxb/mycombn(nn)
  
  ARI = (RI - expectNull)/((maxa+maxb)/2-expectNull)
  return(ARI)
}


# b) ----------------------------------------------------------------------

myARI(cellCycle.kmeans$cluster,cellCycle[,79])
adjustedRandIndex(cellCycle.kmeans$cluster,cellCycle[,79])

myARI(clusterHclust,cellCycle[,79])
adjustedRandIndex(clusterHclust,cellCycle[,79])





