## Caleb 2017/09/24 CTRB 5320

setwd("/Users/zhuo/Dropbox/Caleb-Huo.github.io/teaching/data/UScityDistance")

z <- read.delim("Mileage.txt", row.names=1)


z_mds = - cmdscale(dist(z),k=2)

z_sammon = -  MASS::sammon(dist(z),k=2)$points
delta <- 500

png("cityDistance.png")
plot(z_mds[,1],z_mds[,2], type="n", main="MDS", xlab="", ylab="", 
	xlim=c(min(z_mds[,1]) - delta, max(z_mds[,1]) + delta), ylim=c(min(z_mds[,2]) - delta, max(z_mds[,2]) + delta) )
text(z_mds[,1],z_mds[,2], rownames(z_mds),cex=1.5)
dev.off()

png("cityDistance_sammon.png")
plot(z_sammon[,1],z_sammon[,2], type="n", main="MDS sammon", xlab="", ylab="", 
	xlim=c(min(z_sammon[,1]) - delta, max(z_sammon[,1]) + delta), ylim=c(min(z_sammon[,2]) - delta, max(z_sammon[,2]) + delta) )
text(z_sammon[,1],z_sammon[,2], rownames(z_sammon),cex=1.5)
dev.off()
