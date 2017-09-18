#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' author: "Zhiguang Huo (Caleb)"
#' date: "Monday September 18, 2017"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: R graphics ggplot2
#' ---
#' 
#' 
#' 
#' 
#' 
#' ggplot2
#' ===
#' ggplot2 is based on the **grammer of graphics**,
#' the idea that you can build every graph from the same few components:
#' 
#' - data
#' - aesthetic mapping
#' - geometric object
#' - statistical transformations
#' - scales
#' - coordinate system
#' - position adjustments
#' - faceting
#' 
#' ggplot2 cheetsheet:
#' <https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf>
#' 
#' 
#' ggplot2 grammers
#' ===
#' ![](../figure/ggplot-grammar.png)
#' 
#' 
#' ggplot2 usage -- qplot() and ggplot2()
#' ===
#' 
## ------------------------------------------------------------------------
library(ggplot2)

#' 
#' qplot()
#' 
#' - qplot is a shortcut designed to be familiar if you're used to base plot().
#' - It's a convenient wrapper for creating a number of different types of plots.
#' - It's great to produce plots quickly, but not suitable for complex graphics.
#' - qplot(x, y, data, geom = "auto")
#' 
#' mpg data data
#' === 
## ------------------------------------------------------------------------
str(mpg)
head(mpg)

#' 
#' basic scattered plot
#' === 
#' 
## ------------------------------------------------------------------------
qplot(displ, hwy, data = mpg) 
## aesthetic: displ, hwy. data: mpg

#' 
#' scattered plot with color
#' === 
#' 
## ------------------------------------------------------------------------
qplot(displ, hwy, colour = class, data = mpg) 
## aesthetic: displ, hwy, class; data: mpg

#' 
#' scattered plot with shape
#' === 
#' 
## ------------------------------------------------------------------------
mpg_sub <- subset(mpg, class!="suv") ## qplot support a maximum of 6 shapes
qplot(displ, hwy, shape = class, data = mpg_sub) ## aesthetic: displ, hwy, class; data: mpg

#' 
#' scattered plot with geom = "line"
#' ===
#' 
## ------------------------------------------------------------------------
qplot(displ, hwy, data = mpg, geom = "line") 
## aesthetic: displ, hwy
## data: mpg
## Geometries: line

#' 
#' scattered plot with geom = "path"
#' ===
#' 
## ------------------------------------------------------------------------
qplot(displ, hwy, data = mpg, geom = "path") 
## aesthetic: displ, hwy
## data: mpg
## Geometries: path

#' 
#' More about geom -- geom = "boxplot"
#' ===
## ------------------------------------------------------------------------
qplot(class, displ, data = mpg, geom = "boxplot") 
## aesthetic: displ, class
## data: mpg
## Geometries: boxplot

#' 
#' More about geom -- geom = "jitter"
#' ===
## ------------------------------------------------------------------------
qplot(class, displ, data = mpg, geom = "jitter") 
## aesthetic: displ, class
## data: mpg
## Geometries: jitter

#' 
#' More about geom -- geom = c("jitter", "boxplot")
#' ===
## ------------------------------------------------------------------------
qplot(class, displ, data = mpg, geom = c("boxplot", "jitter")) 
## aesthetic: displ, class
## data: mpg
## Geometries: jitter

#' 
#' 
#' More about geom -- geom = "histogram"
#' ===
## ------------------------------------------------------------------------
qplot(displ, data = mpg, geom = "histogram") 
## aesthetic: displ
## data: mpg
## Geometries: histogram

#' 
#' More about geom -- geom = "density"
#' ===
## ------------------------------------------------------------------------
qplot(displ, data = mpg, geom = "density") 
## aesthetic: displ
## data: mpg
## Geometries: density

#' 
#' Facets
#' ===
#' 
## ------------------------------------------------------------------------
qplot(displ, data = mpg, geom = "density", facets = ~class) 
## aesthetic: displ
## data: mpg
## Geometries: density
## facets: ~class

#' 
#' ggplot() - graphics are added up by different layers
#' ===
#' 
#' Compared to qplot(), it's easier to use multiple dataset in ggplot().
#' 
#' - The basic usage is a combination of ggplot(), aes() and some geom function.
#' - ggplot() initializes a ggplot object. It can declare input data and a set
#' of aesthetics.
#' - aes() describes how variables mapped to visual properties.
#' - geom function describe how the data will be visualized (point, line, histogram),
#' - Use "+" to add layers.
#'     - For example, ggplot(data) + aes(x = a, y = b) + geom point().
#' 
#' 
#' Aesthetics
#' ===
#' 
#' - Colour related aesthetics: colour, fill and alpha
#' - Group related aesthetics: group
#' - Differentiation related aesthetics: linetype, size, shape
#' - Position related aesthetics: x, y, xmin, xmax, ymin, ymax, xend, yend
#' 
#' ggplot example
#' ===
## ------------------------------------------------------------------------
ggplot(mpg) + aes(x=displ, y=hwy, size=cyl) + geom_point()

#' 
#' 
#' ggplot: combine layers
#' ===
## ------------------------------------------------------------------------
myggplot <- ggplot(mpg) + aes(x=displ, y=hwy, color=cyl)
myggplot + geom_point()

#' 
#' 
#' ggplot: geom_line by group
#' ===
## ------------------------------------------------------------------------
ggplot(data = mpg, aes(displ, hwy, colour=class)) + geom_point(aes(size=cyl)) + geom_line(aes(group = class))

#' 
#' Geom functions
#' ===
## ------------------------------------------------------------------------
ls(pattern = '^geom_', env = as.environment('package:ggplot2'))

#' 
#' smooth by group 1
#' ===
## ------------------------------------------------------------------------
ggplot(data = mpg, aes(displ, hwy, colour=class)) + geom_point() + geom_smooth(aes(group = class), method="lm", se = F, size = 1) 

#' 
#' smooth by group 2
#' ===
## ------------------------------------------------------------------------
ggplot(data = mpg, aes(displ, hwy, colour=class)) + geom_smooth(aes(group = class), method="lm", se = T, size = 2) 

#' 
#' ggplot() boxplot
#' ===
## ------------------------------------------------------------------------
mpgbox <- ggplot(mpg, aes(class, hwy)) + geom_boxplot(aes(fill=class))
mpgbox

#' 
#' 
#' ggplot() histogram basic
#' ===
## ------------------------------------------------------------------------
ggplot(mpg, aes(x = hwy)) + geom_histogram(binwidth = 3, fill = "steelblue")

#' 
#' ggplot() histogram fill by color
#' ===
## ------------------------------------------------------------------------
ggplot(mpg, aes(x = hwy)) + geom_histogram(binwidth = 3, aes(fill = class))

#' 
#' ggplot() histogram facets by group
#' ===
## ------------------------------------------------------------------------
ggplot(mpg, aes(x = hwy, fill = class)) + geom_histogram(binwidth = 3) + facet_grid(class ~ .)

#' 
#' bar plot with standard deviation
#' ===
#' 
## ------------------------------------------------------------------------
mpgSummary <- data.frame(class = with(mpg, tapply(class, class, unique)), 
  meanDispl = with(mpg, tapply(displ, class, mean)),
                         sdDispl = with(mpg, tapply(displ, class, sd)))

ggplot(mpgSummary, aes(x=class, y=meanDispl, fill=class)) + 
  geom_bar(position=position_dodge(), stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  geom_errorbar(aes(ymin=meanDispl-sdDispl, ymax=meanDispl+sdDispl),
                size=.3,    # Thinner lines
                width=.2,
                position=position_dodge(.9))

#' 
#' facet
#' ===
#' 
#' - facet_grid: lay out panels in a grid.
#' - facet_null: the default setting (single panel).
#' - facet_wrap: wrap a 1d ribbon of panels into proper 2d display
#' 
#' ggplot() histogram facet_wrap
#' ===
#' 
## ------------------------------------------------------------------------
ggplot(mpg, aes(x = hwy, fill = class)) + geom_histogram(binwidth = 3) + facet_wrap(~ class)

#' 
#' facet_null: back to single panel
#' ===
#' 
## ------------------------------------------------------------------------
ggplot(mpg, aes(x = hwy, fill = class)) + geom_histogram(binwidth = 3) + facet_grid(class ~ .)  + facet_null()

#' 
#' bar plot
#' ===
## ------------------------------------------------------------------------
ggplot(mpg, aes(class, fill=as.factor(cyl))) + geom_bar(position="dodge") #side by side

#' 
#' ---
#' 
## ------------------------------------------------------------------------
ggplot(mpg, aes(class, fill=as.factor(cyl))) + geom_bar(position="fill") #fill

#' 
#' ---
#' 
## ------------------------------------------------------------------------
ggplot(mpg, aes(class, fill=as.factor(cyl))) + geom_bar(position="stack") #default

#' 
#' 
#' 
#' Stat transformation
#' ===
#' 
#' - stat ecdf: Empirical Cumulative Density Function.
#' - stat ellipse: Plot data ellipses.
#' - stat function: Superimpose a function.
#' - stat identity: Identity statistic.
#' - stat qq: Calculation for quantile-quantile plot.
#' - stat summary 2d: Bin and summarise in 2d.
#' - stat unique: Remove duplicates.
#' 
#' empirical CDF
#' ===
#' 
## ------------------------------------------------------------------------
df <- data.frame(x = rnorm(1000))
ggplot(df, aes(x)) + stat_ecdf(geom = "step")

#' 
#' 
#' ---
#' 
## ------------------------------------------------------------------------
n <- 100
df <- data.frame(x = c(rnorm(n, 0, 3), rnorm(n, 0, 10)),
                 g = gl(2, n))
ggplot(df, aes(x, colour = g)) + stat_ecdf()

#' 
#' stat_function
#' ===
#' 
## ------------------------------------------------------------------------
n <- 100
set.seed(32611)
df <- data.frame(
  x = rnorm(n)
)
x <- df$x
base <- ggplot(df, aes(x)) + geom_density()
base + stat_function(fun = dnorm, colour = "red") + xlim(c(-3,3))

#' 
#' stat_ellipse
#' ===
#' 
## ------------------------------------------------------------------------
ggplot(mpg, aes(x = displ, y = hwy)) + geom_point() +   stat_ellipse()

#' 
#' 
#' stat_ellipse by group
#' ===
#' 
## ------------------------------------------------------------------------
ggplot(mpg, aes(x = displ, y = hwy, color=displ > 4)) + geom_point() +   stat_ellipse()

#' 
#' 
#' Coordinate
#' ===
#' 
#' - coord_cartesian: Cartesian coordinates.
#' - coord_fixed: control_fixed ratio between x and y scales.
#' - coord_flip: flip x and y.
#' - coord_map: map projections.
#' - coord_polar: polar coordinates.
#' - coord_trans: transformed cartesian coordinate system.
#' 
#' Our old friend mpg
#' ===
#' 
## ------------------------------------------------------------------------
p <- ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  geom_smooth()

p

#' 
#' ---
#' 
#' # Setting the limits on the coordinate system performs a visual zoom.
## ------------------------------------------------------------------------
p + coord_cartesian(xlim = c(3, 5), expand = FALSE)

#' 
#' ---
#' 
#' Setting the limits on a scale converts all values outside the range to NA.
## ------------------------------------------------------------------------
p + xlim(3, 5)
#the same as p + scale_x_continuous(limits = c(325, 500))

#' 
#' 
#' 
#' resize the plot
#' ===
#' 
## ------------------------------------------------------------------------
p <- ggplot(mpg, aes(displ, hwy)) +  geom_point()
p + coord_fixed(ratio = 0.5)
p + coord_fixed(ratio = 0.1)

#' 
#' flip x and y
#' ===
#' 
## ------------------------------------------------------------------------
ggplot(mpg, aes(class, hwy)) +
  geom_boxplot() +
  coord_flip()

#' 
#' draw a world map
#' ===
#' 
## ------------------------------------------------------------------------
world <- map_data("world")
worldmap <- ggplot(world, aes(x=long, y=lat, group=group)) +
  geom_path() +
  scale_y_continuous(breaks = (-2:2) * 30) +
  scale_x_continuous(breaks = (-4:4) * 45)

worldmap


#' 
#' ---
#' 
## ------------------------------------------------------------------------
worldmap + coord_map("ortho", orientation = c(41, -74, 0))


#' 
#' polar coordinte for pie chart
#' ===
#' 
## ------------------------------------------------------------------------
pie <- ggplot(mpg, aes(x = factor(1), fill = factor(cyl))) +
  geom_bar(width = 1)
pie + coord_polar(theta = "y")

#' 
#' 
#' 
#' Theme
#' ===
#' 
#' - theme_grey: signature ggplot2 theme.
#' - theme_bw:dark-on-light theme.
#' - theme_linedraw: flip x and y.
#' - theme_light: light grey background.
#' - theme_dark: dark background.
#' - theme_minimal: no background annotations.
#' - theme_classic: classic-looking theme.
#' - theme_void: empty theme.
#' 
#' examples on different themes
#' ===
## ------------------------------------------------------------------------
p <- ggplot(mpg) + geom_point(aes(x = displ, y = hwy,
                                     colour=factor(cyl))) + facet_wrap(~class)
p

#' 
#' 
#' ---
#' 
## ------------------------------------------------------------------------
p + theme_bw()

#' 
#' More about theme
#' ===
#' 
## ------------------------------------------------------------------------
p <- ggplot(mpg) + geom_point(aes(x = displ, y = hwy,
                                     colour=factor(cyl))) 
p + theme(text = element_text(size=20),
        axis.text.x = element_text(angle=90, hjust=1)) 


#' 
#' labels
#' ===
#' 
## ------------------------------------------------------------------------
p <- ggplot(mpg) + geom_point(aes(x = displ, y = hwy,
                                     colour=factor(cyl))) 
p + 
  labs(title = "New plot title", x = "New x label", y = "New y label")

## more please check ?labs

#' 
#' ---
#' 
## ------------------------------------------------------------------------

p + labs(title = "New plot title", x = "New x label", y = "New y label")+ theme(axis.text.x = element_text(family = "Arial", colour = "red")) 

## more please check ?theme

#' 
#' How to learn more about ggplot2
#' ===
#' 
#' 1. ggplot2 cheetsheet: <https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf>
#' 2. google
#' 
#' 
## ------------------------------------------------------------------------
knitr::purl("Rgraph_ggplot2.rmd", output = "Rgraph_ggplot2.R ", documentation = 2)

