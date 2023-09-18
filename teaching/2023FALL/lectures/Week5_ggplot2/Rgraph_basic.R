#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Tuesday September 19, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: R graphics
#' ---
#' 
#' 
#' Plotting in R
#' ===
#' 
#' - plot(): a generic plotting function
#'     - points(): add points to the plot.
#'     - lines(), abline(): add lines
#'     - text(): add text 
#'     - legend(): add legend
#' - curve(): any function curve
#' - hist(): histogram
#' - boxplot(): boxplot
#' - contour(): contour plot
#' - image(), heatmap(): heatmaps
#' 
#' 
#' 
#' scatter plot
#' ===
## ---------------------------------------------------------------------------------------------
n <- 80
set.seed(32611)
x <- sort(runif(n, min = 0, max=2*pi))
y <- sin(x) + rnorm(n, mean = 0, sd = 0.2)
plot(x,y)

#' 
#' 
#' Type
#' ===
#' 
#' - "p": for points
#' - "l": for lines
#' - "b" for both
#' - "o" for both ‘overplotted’
#' - "n" for no plotting
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
plot(x,y, type="p") ## default is "p"

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
plot(x,y, type="l") ## # default xlab and ylab are the variable

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
plot(x,y, type="b")

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
plot(x,y, type="o")

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
plot(x,y, type="n")

#' 
#' 
#' Labels
#' ===
#' 
#' - main: title
#' - xlab: x labels
#' - ylab: y labels
#' 
## ---------------------------------------------------------------------------------------------
plot(x,y, main="sin function with Gaussian noise", 
     xlab="x axis", ylab="y axis") 

#' 
#' Text and Symbol Size
#' ===
#' - cex: number indicating the amount by which plotting text and symbols should be scaled relative to the default. 1=default, 1.5 is 50% larger, 0.5 is 50% smaller, etc.
#' - cex.axis	magnification of axis annotation relative to cex
#' - cex.lab	magnification of x and y labels relative to cex
#' - cex.main	magnification of titles relative to cex
#' - cex.sub	magnification of subtitles relative to cex
#' 
#' --- 
## ---------------------------------------------------------------------------------------------
plot(x,y, main="sin function with Gaussian noise", 
     xlab="x axis", ylab="y axis", 
     cex=2, cex.axis=2, cex.lab=2, cex.main=2, cex.sub=2) 

#' 
#' 
#' Text
#' ===
#' 
#' - After create a plot, you can add text to it by text.
#' 
## ---------------------------------------------------------------------------------------------
letters[1:10] ## letters contain a-z
plot(x=1:10, y=1:10, type="n", main = "some texts") 
text(x = 1:10, y=1:10, labels = letters[1:10])

#' 
#' Text (2)
#' ===
#' 
#' - text position
#'   - 1: bottom
#'   - 2: left
#'   - 3: top
#'   - 4: right
#'   - NULL: original (default)
#' 
## ---------------------------------------------------------------------------------------------
letters[1:10] ## letters contain a-z
plot(x=1:10, y=1:10, main = "some texts") 
text(x = 1:10, y=1:10, labels = letters[1:10], pos = NULL)

#' 
#' 
#' 
#' Math expression
#' ===
## ---------------------------------------------------------------------------------------------

plot(1:10, type="n", xlab="X", ylab="Y")
text(5.5, 9, "expression(y==alpha[1]*x+alpha[2]*x^2)", cex=1.5)
text(5.5, 8, expression(y==alpha[1]*x+alpha[2]*x^2), cex=1.5)
theta = 3
text(5.5, 6, "bquote(hat(theta)==.(theta))", cex=1.5)
text(5.5, 5, bquote(hat(theta)==.(theta)), cex=1.5)


#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' Point type
#' ===
#' - pch is an argument for point typs
## ---------------------------------------------------------------------------------------------
plot(1:25,1:25,pch=1:25)

#' 
#' ---
#' 
#' - use pch = 19
## ---------------------------------------------------------------------------------------------
plot(x,y, pch=19) 

#' 
#' 
#' 
#' 
#' Multiple plots
#' ===
#' - Use par() with mfcol or mfrow settings to create a nxm grid of figures.
#'     - mfcol=c(nr, nc) adds figures by column
#'     - mfrow=c(nr, nc) adds figures by row
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
n <- 80
set.seed(32611)
x <- sort(runif(n, min = 0, max=2*pi))
y1 <- sin(x) + rnorm(n, mean = 0, sd = 0.2)
y2 <- cos(x) + rnorm(n, mean = 0, sd = 0.2)

par(mfrow=c(1,2))
plot(x,y1, main="sin plot")
plot(x,y2, main="cos plot")

#' 
#' Lines
#' ===
#' - lty:	line type. see the chart below.
#' 
#' ![](../figure/lines.png)
#' 
#' - lwd	line width relative to the default (default=1). 2 is twice as wide.
#' 
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
par(mfrow=c(2,2))
plot(x,y, type="l",lwd=1) 
plot(x,y, type="l",lwd=2) 
plot(x,y, type="l",lwd=3) 
plot(x,y, type="l",lwd=4) 

#' 
#' 
#' abline
#' ===
#' 
#' - a: intercept 
#' - b: slope
#' 
## ---------------------------------------------------------------------------------------------
plot(0:6, 0:6, type = "n")
abline(a = 3, b = -1)

#' 
#' 
#' abline
#' ===
#' 
#' - h:	the y-value(s) for horizontal line(s).
#' - v: the x-value(s) for vertical line(s).
#' 
## ---------------------------------------------------------------------------------------------
plot(0:6, 0:6, type = "n")
abline(h = 3)
abline(v = 3, col=2)

#' 
#' 
#' abline
#' ===
#' 
#' - reg: an object with a coef method. 
#' - coef: a vector of length two giving the intercept and slope.
#' 
## ---------------------------------------------------------------------------------------------
z <- lm(dist ~ speed, data = cars)
plot(cars)
abline(z) # equivalent to abline(reg = z) or
# abline(coef = coef(z))

#' 
#' Color
#' ===
#' - col argument to control the color
#' - col = 1:8 for eight basic colors
#' - a string for 657 available colors (check out colors() )
#' - RGB: "#rrggbb". 
#'     - rr: red intensity, 00-FF (Hexadecimal)
#'     - gg: green intensity, 00-FF (Hexadecimal)
#'     - bb: blue intensity, 00-FF (Hexadecimal)
#'     - e.g. #FF0000 represent red color.
#'     
#' ---
## ---------------------------------------------------------------------------------------------
plot(1:8,1:8, col=1:8) 

#' 
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
par(mfrow=c(2,2))
plot(x,y, main = "black color") ## default is black
plot(x,y, main = "blue color", col=4) ## basic color option 1:8
plot(x,y, main = "green color", col='green') ## find colors from colors()
plot(x,y, main = "red color", col='#FF0000') ## RGB mode

#' 
#' More on colors
#' ===
#' - More on color argument for plot
#'     - col	plotting color. 
#'     - col.axis	color for axis annotation
#'     - col.lab	color for x and y labels
#'     - col.main	color for titles
#'     - col.sub	color for subtitles
#'     - fg	plot foreground color (axes, boxes - also sets col= to same)
#'     - bg	plot background color
#' - You can also create a vector of n contiguous colors using the functions rainbow(n), heat.colors(n), terrain.colors(n), topo.colors(n), and cm.colors(n).
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
n <- 10
par(mfrow=c(2,3))
plot(1:n,1:n, col=rainbow(n), main = "rainbow") 
plot(1:n,1:n, col=heat.colors(n), main = "heat.colors") 
plot(1:n,1:n, col=terrain.colors(n), main = "terrain.colors") 
plot(1:n,1:n, col=topo.colors(n), main = "topo.colors") 
plot(1:n,1:n, col=cm.colors(n), main = "cm.colors") 

#' 
#' 
#' 
#' 
#' 
#' Curve
#' ===
#' 
#' - curve() allows you to plot any function
## ---------------------------------------------------------------------------------------------
curve(sin, from = 0, to = 2*pi)

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
chippy <- function(x) sin(cos(x)*exp(-x/2))
curve(chippy, -8, 7, n = 2001) # n specify number of points used to draw the curve, default n = 101

#' 
#' Legend
#' ===
#' 
#' - ?legend for more options
#'  
## ---------------------------------------------------------------------------------------------
curve(sin, 0, 2*pi, col=2)
curve(cos, 0, 2*pi, col=4, add=T) ## add=T to overlay a new curve on top of the original figure
legend("bottomleft", legend = c("sin", "cos"), col = c(2,4),lty = 1)

#' 
#' 
#' segments
#' ===
#' 
#' - add line segment
#' 
## ---------------------------------------------------------------------------------------------
plot(1:3,1:3,col = "blue")
segments(x0 = 3,y0 = 3,x1 = 1,y1 = 1,lwd=2,col=2)

#' 
#' 
#' arrows
#' ===
#' 
#' - add arrows
#' 
## ---------------------------------------------------------------------------------------------
plot(1:3,1:3,col = "blue")
arrows(x0 = 3,y0 = 3,x1 = 1,y1 = 1,code=1,angle=30,length=.1,lwd=2)

#' 
#' polygon
#' ===
#' 
#' - add polygon
#' 
## ---------------------------------------------------------------------------------------------
plot(1:3,1:3,col = "blue")
polygon(x = c(2,3,3,2),y = c(2,2,3,3),col="pink")

#' 
#' A comprehensive example
#' ===
## ---------------------------------------------------------------------------------------------
set.seed(32611)
curve(dnorm,-4,4, ylim = c(-0.1, 0.5)) 

# shaded area
xvals <- seq(-4,2,length=100) 
dvals <- dnorm(xvals)
polygon(c(xvals,rev(xvals)),c(rep(0,100),rev(dvals)),col="gray")

#label pnorm as area under the curve 
arrows(1,.15,2,.25,code=1,angle=30,length=.1,lwd=2)
text(2,.25,paste('pnorm(2) =',round(pnorm(2),3)),cex=.75,pos=3)

#label dnorm as height
segments(2,0,2,dnorm(2),lwd=2,col=2)
arrows(2,.025,2.5,.1, code=1, angle=30, length=.1, lwd=2, col=2)
text(2.5,.1,paste('dnorm(2) =', round(dnorm(2),3)), cex=.75, pos=3, col=2)

#label qnorm as quantile
points(2,0,col=4,pch=16,cex=1.1)
arrows(2,0,3,.05,code=1,angle=30,length=.1,lwd=2,col=4)
text(3,.05,paste('qnorm(',round(pnorm(2),3),') = 2'), cex=.75,pos=3,col=4)

mtext(side=3,line=.5,'X ~ Normal(0,1)',cex=.9,font=2)
points(rnorm(20),jitter(rep(0,20)),pch=18,cex=.9)
legend(-4,.3,'rnorm(20)',pch=18,cex=.8,bty='n')

#' 
#' margin
#' ===
#' ![](../figure/Margin.png)
#' 
#' - Margin order: bottom, left, top, right
#' 
#' 
#' ---
#' 
#' - to change margins, use the par() function, with the argument mar. 
## ---------------------------------------------------------------------------------------------
par(mar=c(1,2,3,4)) # bottem, left, top, right order, inner margin
plot(x, y, main="Red sin", pch=20, col="red")

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
par(oma = c(4,3,2,1), mar=c(1,2,3,4)) # bottem, left, top, right order, inner margin and outer margin
plot(x, y, main="Red sin", pch=20, col="red")

#' 
#' mtext
#' ===
#' 
#' - add text to margins
#' - side
#'   - 1: bottom
#'   - 2: left
#'   - 3: top
#'   - 4: right
#' 
## ---------------------------------------------------------------------------------------------
plot(1:10, type="n", xlab="X", ylab="Y")
mtext(text = "mtext", side = 4, line = 1, col="red")

#' 
#' 
#' save plot
#' ===
#' - Figures can be saved as .pdf, .png, .jpeg, .bmp, .tiff by using pdf(), png(), jpeg(), bmp(), tiff().
#' - PDF format is recommended since it is vector format. resize the figure woun't affect the quality.
#' 
## ---------------------------------------------------------------------------------------------
n <- 80
set.seed(32611)
x <- sort(runif(n, min = 0, max=2*pi))
y <- sin(x) + rnorm(n, mean = 0, sd = 0.2)

pdf("sinFunction.pdf")
plot(x,y)
dev.off()

#' 
#' 
#' IRIS data example
#' ===
## ---------------------------------------------------------------------------------------------
data(iris)
head(iris)
str(iris)

#' 
#' boxplot
#' ===
## ---------------------------------------------------------------------------------------------
boxplot(Petal.Length ~ Species, data = iris)

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
names(iris)
par(mfrow = c(2,2))
for(i in 1:4){
  aname <- names(iris)[i]
  boxplot(iris[[i]] ~ iris$Species, xlab="Species", col=i, ylab=aname, main=aname)  
}

#' 
#' 
#' histogram
#' ===
#' 
#' - default
#' - set nclass
#' - density
#' 
## ---------------------------------------------------------------------------------------------
par(mfrow=c(2,2))
hist(iris$Petal.Length) ## default
hist(iris$Petal.Length, nclass=50) ## set number of bins
hist(iris$Petal.Length,  prob=TRUE, col="grey") ## grey color
lines(density(iris$Petal.Length), col="blue") ## and put the density function on top of the histogram

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
par(mfrow = c(2,2))
xlims <- range(iris$Petal.Length) ## set common x limits
uniqueSpecies <- levels(iris$Species)
for(i in seq_along(uniqueSpecies)){
  aspecies <- uniqueSpecies[i]
  sampleSelection <- iris$Species==aspecies
  adata <- iris$Petal.Length[sampleSelection]
  hist(adata, col=i, xlim=xlims, xlab="petal length",main=aspecies)
}


#' 
#' 
#' contour plot
#' ===
## ---------------------------------------------------------------------------------------------
x <- y <- seq(-1, 1, len=25)
z <- outer(x, y, FUN=function(x,y) -x*y*exp(-x^2-y^2))
# Contour plots
contour(x,y,z, main="Contour Plot")
filled.contour(x,y,z, main="Filled Contour Plot")
filled.contour(x,y,z, color.palette = heat.colors)
filled.contour(x,y,z, color.palette = colorRampPalette(c("red", "white", "blue")))

#' 
#' 
#' 
#' Heatmap.2
#' ===
#' 
#' - A tutorial: https://sebastianraschka.com/Articles/heatmaps_in_r.html.
#' 
## ---------------------------------------------------------------------------------------------
library(gplots) ## heatmap.2 is available in gplots
?heatmap.2

#' 
#' 
#' Application on Iris data
#' ===
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
heatmap.2(dataMatrix)

#' 
#' ---
#' 
#' - Remove level trace
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
heatmap.2(dataMatrix, trace = "none")

#' 
#' 
#' ---
#' 
#' - Remove level trace
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix2 <- as.matrix(iris[1:5,1:4])
heatmap.2(dataMatrix2, trace = "none", 
          cellnote = dataMatrix2, notecol = "black")

#' 
#' 
#' ---
#' 
#' - Put colSideColorBar
#'     - RowSideColors
#'     - legend
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
species <- iris$Species
color0 <- species
levels(color0) <- palette()[1:length(levels(species))]
color <- as.character(color0)

heatmap.2(dataMatrix, trace = "none", RowSideColors = color)
legend("topright", legend=levels(species),fill=levels(color0))

#' 
#' ---
#' 
#' - Remove automatical reordering of rows and columns
#'   - Colv = NA
#'   - Rowv = NA
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
species <- iris$Species
color0 <- species
levels(color0) <- palette()[1:length(levels(species))]
color <- as.character(color0)

heatmap.2(dataMatrix, trace = "none", RowSideColors = color,
           Colv=NA, Rowv = NA )
legend("topright", legend=levels(species),fill=levels(color0))

#' 
#' ---
#' 
#' - Standardize each row to mean 0 and sd 1
#'   - scale = "row"
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
species <- iris$Species
color0 <- species
levels(color0) <- palette()[1:length(levels(species))]
color <- as.character(color0)

heatmap.2(dataMatrix, trace = "none", RowSideColors = color,
           Colv=NA, Rowv = NA , scale = "row")
legend("topright", legend=levels(species),fill=levels(color0))

#' 
#' 
#' ---
#' 
#' - Try other colors
#'   - heat.colors (default)
#'   - redgreen
#'   - greenred
#'   - grey.colors
#'   - bluered
#'   - cm.colors
#'   - rainbow
#'   - terrain.colors
#'   - topo.colors
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
species <- iris$Species
color0 <- species
levels(color0) <- palette()[1:length(levels(species))]
color <- as.character(color0)

heatmap.2(dataMatrix, trace = "none", 
          RowSideColors = color, col = bluered, 
           Colv=NA, Rowv = NA , scale = "row")
legend("topright", legend=levels(species),fill=levels(color0))

#' 
#' 
#' Adjust legend positions
#' ===
#' 
## ---------------------------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
species <- iris$Species
color0 <- species
levels(color0) <- palette()[1:length(levels(species))]
color <- as.character(color0)

heatmap.2(dataMatrix, trace = "none", 
          RowSideColors = color, col = bluered, 
           Colv=NA, Rowv = NA , scale = "row", 
          margins = c(6,6))
par(xpd=TRUE) ## default is FALSE, allow legend outside the plotting area
legend(x = 0.7, y = 1.1, legend=levels(species),fill=levels(color0))

#' 
#' 
#' Advanced heatmap 
#' ===
#' 
#' - ComplexHeatmap
#'     - Check out the tutorial <https://jokergoo.github.io/ComplexHeatmap-reference/book/>
