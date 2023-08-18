#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday September 19, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: R graphics heatmap.2
#' ---
#' 
#' 
#' 
#' Heatmap.2
#' ===
#' 
#' - A tutorial: https://sebastianraschka.com/Articles/heatmaps_in_r.html.
#' 
## -----------------------------------------------------------------------------
library(gplots) ## heatmap.2 is available in gplots
?heatmap.2

#' 
#' 
#' Application on Iris data
#' ===
#' 
## -----------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
heatmap.2(dataMatrix)

#' 
#' ---
#' 
#' - Remove level trace
#' 
## -----------------------------------------------------------------------------
dataMatrix <- as.matrix(iris[,1:4])
heatmap.2(dataMatrix, trace = "none")

#' 
#' 
#' ---
#' 
#' - Remove level trace
#' 
## -----------------------------------------------------------------------------
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
## -----------------------------------------------------------------------------
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
## -----------------------------------------------------------------------------
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
## -----------------------------------------------------------------------------
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
## -----------------------------------------------------------------------------
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
## -----------------------------------------------------------------------------
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
