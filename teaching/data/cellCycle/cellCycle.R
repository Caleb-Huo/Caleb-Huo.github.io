setwd("~/Dropbox/Caleb-Huo.github.io/teaching/data/cellCycle")
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
varianceExplained <- PCA_cellCycle_gene$sdev^2/sum(PCA_cellCycle_gene$sdev^2)
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
PCValues$label <- 0
for(aphase in PCValues$phase){
	aselection <- PCValues$phase == aphase
	PCValues$label[aselection] <- 1:sum(aselection)
}


varianceExplained <- PCA_cellCycle_sample$sdev^2/sum(PCA_cellCycle_sample$sdev^2)
varianceExplained_PC1 <- paste0("variance explained: ",signif(varianceExplained[1]* 100,3),"%")
varianceExplained_PC2 <- paste0("variance explained: ",signif(varianceExplained[2]* 100,3),"%")

subsetPhase <- PCValues[- grep("cl", PCValues$phase), ]

png("cellCycleBySamples.png")
ggplot(subsetPhase, aes(x=PC1, y=PC2, color=phase, label=label)) + geom_path() + geom_text() + 
facet_wrap(~phase)+ labs(x = varianceExplained_PC1, y = varianceExplained_PC2) +
theme(text = element_text(size=20)) 
dev.off()



