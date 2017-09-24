library(ggplot2)
library(ggrepel)

setwd("~/Dropbox/Caleb-Huo.github.io/teaching/data/SNP_MDS")

z0 <- read.csv("SNP_PCA.csv", row.names=1)
z <- as.dist(cbind(z0,NA))

z_mds = - cmdscale(dist(z),k=2)

mdsDataFrame <- data.frame(Race=rownames(z_mds),x = z_mds[,1], y = z_mds[,2])

png("population_SPN.png")
ggplot(data=mdsDataFrame, aes(x=x, y=y, label=Race)) + geom_point(aes(color=Race)) + geom_text_repel(data=mdsDataFrame, aes(label=Race))
dev.off()
