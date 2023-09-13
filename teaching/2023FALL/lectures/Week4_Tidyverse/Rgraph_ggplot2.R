#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Thursday September 14, 2023"
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
#' 
#' ggplot2 grammers
#' ===
#' ![](../figure/ggplot-grammar.png){width=50%}
#' 
#' ggplot() - graphics are added up by different layers
#' ===
#' 
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
#' Aesthetics --- aes()
#' ===
#' 
#' - x variable and y variable
#' - Colour related aesthetics: colour, fill and alpha
#' - Group related aesthetics: group
#' - Differentiation related aesthetics: linetype, size, shape
#' - Position related aesthetics: x, y, xmin, xmax, ymin, ymax, xend, yend
#' 
#' 
#' load ggplot2 package
#' === 
#' 
## -----------------------------------------------------------------------------
library(ggplot2) ## part of tidyverse
library(tidyverse)

#' 
#' mpg data data
#' === 
## -----------------------------------------------------------------------------
str(mpg)
head(mpg)

#' 
#' 
#' ggplot example
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + aes(x=displ, y=hwy) + geom_point()

#' 
#' 
#' ggplot: combine layers
#' ===
## -----------------------------------------------------------------------------
myggplot <- ggplot(data = mpg) + aes(x=displ, y=hwy)
myggplot + geom_point()

#' 
#' aes -- color (continuous)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x=displ, y=hwy, color = cyl) +
  geom_point()

#' 
#' aes -- color (categorical)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x=displ, y=hwy, color = class) +
  geom_point()

#' 
#' 
#' aes -- color (absolute color)
#' ===
## ---- eval = FALSE------------------------------------------------------------
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy, color = "blue") +
##   geom_point() ## Doesn't work, aes only maps a variable (in the data) to a color.
## 
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy, color = I("blue")) +
##   geom_point() ## use I to indicate absolute color
## 
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy ) +
##   geom_point(color = "blue") ## or use the color here (outside aes)

#' 
#' 
#' 
#' aes -- color (categorical)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x=displ, y=hwy, color = class) +
  geom_point()

#' 
#' 
#' aes -- size 
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x=displ, y=hwy, size = cyl) +
  geom_point()

#' 
#' 
#' aes -- size (absolute size)
#' ===
## ---- eval = FALSE------------------------------------------------------------
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy, size = "3") +
##   geom_point() ## Doesn't work, aes only maps a variable (in the data) to a size
## 
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy, size = I(3)) +
##   geom_point() ## use I to indicate absolute size
## 
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy ) +
##   geom_point(size = 3) ## or use the size here

#' 
#' 
#' aes -- alpha (transparency)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x=displ, y=hwy, alpha = cyl) +
  geom_point()

#' 
#' 
#' aes -- alpha (absolute alpha)
#' ===
#' 
#' - alpha
#'   - 0: transparent
#'   - 1: solid
#' 
## ---- eval = FALSE------------------------------------------------------------
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy, alpha = 1) +
##   geom_point() ## Doesn't work, aes only maps a variable (in the data) to a alpha
## 
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy, alpha = I(0.5)) +
##   geom_point() ## use I to indicate absolute alpha
## 
## ggplot(data = mpg) +
##   aes(x=displ, y=hwy ) +
##   geom_point(alpha = 0.5) ## or use the alpha here

#' 
#' 
#' 
#' aes -- shape 
#' ===
## -----------------------------------------------------------------------------
mpg_sub <- subset(mpg, class!="suv")
ggplot(data = mpg_sub) + 
  aes(x=displ, y=hwy, shape = class) +
  geom_point()

#' 
#' 
#' ---
#' 
#' - using pipe
#' 
## -----------------------------------------------------------------------------
mpg %>% 
  filter(class!="suv") %>%
  ggplot() + 
  aes(x=displ, y=hwy, shape = class) +
  geom_point()

#' 
#' aes by variable names
#' ===
## -----------------------------------------------------------------------------
xvariable = "displ"
yvariable = "hwy"

ggplot(data = mpg) + 
  aes_string(x=xvariable, y=yvariable, color = "class") +
  geom_point()

#' 
#' 
#' Geom functions
#' ===
## -----------------------------------------------------------------------------
ls(pattern = '^geom_', env = as.environment('package:ggplot2'))

#' 
#' ggplot: geom_line by group
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy, colour=class) + 
  geom_point() + 
  geom_line()

#' 
#' 
#' ---
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy) + 
  geom_point() + 
  geom_line(aes(colour=class))

#' 
#' 
#' ggplot: aes()
#' ===
#' 
#' - global aes: will be applied to each of the geom function
## ---- eval=FALSE--------------------------------------------------------------
## ggplot(data = mpg) +
##   aes(displ, hwy, colour=class) + ## this is global color
##   geom_point() +
##   geom_line()

#' 
#' - local aes: will be used locally
## ---- eval=FALSE--------------------------------------------------------------
## ggplot(data = mpg) +
##   aes(displ, hwy) +
##   geom_point() +
##   geom_line(aes(colour=class)) ## this is local color

#' 
#' 
#' Line segments
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy, colour = class) + 
  geom_point() + 
  geom_abline(aes(intercept = 0, slope = 5), color = "green") + 
  geom_hline(aes(yintercept = 30), color = "blue") + 
  geom_vline(aes(xintercept = 5), color = "red") 

#' 
#' 
#' 
#' smooth by group 1 
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy) + 
  geom_point(aes(colour=class)) + 
  geom_smooth() 

#' 
#' 
#' smooth by group 2
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy) + 
  geom_point(aes(colour=class)) + 
  geom_smooth(method="lm") 

#' 
#' smooth by group 3
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy) + 
  geom_point(aes(colour=class)) + 
  geom_smooth(aes(group=class), method="lm") 

#' 
#' 
#' smooth by group 4
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy, colour = class) + ## global aes will be applied to all higher level aes
  geom_point() + 
  geom_smooth(method="lm") 

#' 
#' smooth by group 5
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(displ, hwy, colour = class) + ## lower level aes will be applied to all higher level aes
  geom_point() + 
  geom_smooth(method="lm", se = F, size = 2) 

#' 
#' 
#' 
#' ggplot() boxplot
#' ===
## -----------------------------------------------------------------------------
mpgbox <- ggplot(data = mpg) + 
  aes(class, hwy) + 
  geom_boxplot(aes(fill=class))
mpgbox

#' 
#' 
#' ggplot() jitter
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(class, hwy, color=class) + 
  geom_jitter()

#' 
#' 
#' ggplot() boxplot + jitter
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(class, hwy, color=class) + 
  geom_boxplot() + 
  geom_jitter()

#' 
#' - Also try this
## ---- eval = FALSE------------------------------------------------------------
## ggplot(data = mpg) +
##   aes(class, hwy, color=class) +
##   geom_jitter() +
##   geom_boxplot()

#' 
#' - ggplot layers are of the same order of your code
#' 
#' 
#' ggplot() violin plot
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(class, hwy, fill=class) + 
  geom_violin() 

#' 
#' - wider parts indicate larger number of counts
#' 
#' 
#' ggplot() bar plot 1
#' ===
#' 
#' - geom_bar(), by default is for counting. So only need aes(x = ...) 
#' 
## -----------------------------------------------------------------------------
ggplot(mpg) + 
  aes(class) + 
  geom_bar()

#' 
#' 
## ---- eval = FALSE------------------------------------------------------------
## ggplot(mpg) +
##   aes(class, color = class) +
##   geom_bar()

#' 
#' - color for geom_bar is for the border color
#' - use fill option to fill in colors
#' 
#' ggplot() bar plot 2
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(mpg) + 
  aes(class, fill=as.factor(cyl)) + 
  geom_bar()

#' 
#' ggplot() bar plot 3
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(mpg) + 
  aes(class, fill=as.factor(cyl)) + 
  geom_bar(position="dodge")  #side by side

#' 
#' ggplot() bar plot: how to specify error bar 1
#' ===
#' 
#' - geom_bar(), by stat="identity" option, we can specify both x and y. 
#' 
#' 
## -----------------------------------------------------------------------------
mpgSummary <- mpg %>%
  group_by(class) %>%
  summarize(meanDispl = mean(displ), sdDispl = sd(displ)) ## sd is standard deviation, standard error se = sd/sqrt(n)

ggplot(data = mpgSummary) + 
  aes(x=class, y=meanDispl, fill=class) + 
  geom_bar(stat="identity",
           colour="black", # Use black outlines,
           size=.3)       # Thinner lines

#' 
#' ggplot() bar plot: how to specify error bar 2
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(data = mpgSummary) + 
  aes(x=class, y=meanDispl, fill=class) + 
  geom_bar(stat="identity",
           colour="black", # Use black outlines,
           size=.3) +      # Thinner lines
  geom_errorbar(aes(ymin=meanDispl-sdDispl, ymax=meanDispl+sdDispl),
                size=.3,    # Thinner lines
                width=.2
                )

#' 
#' ggplot() histogram simple example
#' ===
#' 
#' - geom_histogram(), by default is for counting. So only need aes(x = ...) 
#' 
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x = hwy) + 
  geom_histogram()

#' 
#' 
#' ggplot() histogram fill by color
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x = hwy) + 
  geom_histogram(aes(fill = class))

#' 
#' ggplot() histogram facets by group (1)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x = hwy) + 
  geom_histogram(aes(fill = class)) + 
  facet_wrap(~ class)

#' 
#' 
#' 
#' facet
#' ===
#' 
#' - facet_grid: lay out panels in a grid.
#' - facet_wrap: wrap a 1d ribbon of panels into proper 2d display
#' 
#' 
#' ggplot() histogram facets by group (2)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x = hwy) + 
  geom_histogram(aes(fill = class)) + 
  facet_grid(. ~ class) ## or facet_grid(cols = vars(class))


#' 
#' 
#' ggplot() histogram facets by group (3)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x = hwy) + 
  geom_histogram(aes(fill = class)) + 
  facet_grid(class ~ .) ## or facet_grid(rows = vars(class))

#' 
#' ggplot() histogram facets by group (4)
#' ===
## -----------------------------------------------------------------------------
ggplot(data = mpg) + 
  aes(x = hwy) + 
  geom_histogram(aes(fill = class)) + 
  facet_grid(drv ~ class) ## or facet_grid(rows = vars(drv), cols = vars(class))

#' 
#' 
#' longitudinal data visualization
#' ===
#' 
#' - spaghetti plot
#' - individual trajectory
#' - mean trajectory (with se bar)
#' 
#' sleepstudy: Reaction times in a sleep deprivation study
#' ===
#' 
#' - This laboratory experiment measured the effect of sleep deprivation on cognitive performance.
#' - There were 18 subjects, chosen from the population of interest (long-distance truck drivers), in the 10 day trial. These subjects were restricted to 3 hours sleep per night during the trial.
#' - On each day of the trial each subject’s reaction time was measured. The reaction time shown here is the average of several measurements.
#' - These data are balanced in that each subject is measured the same number of times and on the same occasions.
#' 
#' sleepstudy: Reaction times in a sleep deprivation study
#' ===
#' 
## -----------------------------------------------------------------------------
library(lme4)
data(sleepstudy)
head(sleepstudy, n=5)

#' 
#' spaghetti plot
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(data=sleepstudy) + 
  aes(x = Days, y=Reaction, colour = Subject) +
  geom_path()

#' 
#' individual subject lm smooth
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(data=sleepstudy) + 
  aes(x = Days, y=Reaction, colour = Subject) +
  geom_smooth(method="lm") + 
  facet_wrap(~Subject)

#' 
#' mean trajectory (with SE bar)
#' ===
#' 
## -----------------------------------------------------------------------------
sleepSummary <- sleepstudy %>% 
  group_by(Days) %>%
  summarize(Mean = mean(Reaction), SD = sd(Reaction), SE = sd(Reaction)/sqrt(n()))

ggplot(data=sleepSummary) + 
  aes(x = Days, y=Mean) +
  geom_path() + 
  geom_errorbar(aes(ymin=Mean-SE, ymax=Mean+SE),
                  size=0.5,    # Thinner lines
                  width=.2) 

#' 
#' Add text annotations to a graph
#' ===
#' 
#' - Ref: http://www.sthda.com/english/wiki/ggplot2-texts-add-text-annotations-to-a-graph-in-r-software
#'   - geom_text(): adds text directly to the plot
#'   - geom_label(): draws a rectangle underneath the text, making it easier to read.
#'   - annotate(): useful for adding small text annotations at a particular location on the plot
#'   - annotation_custom(): Adds static annotations that are the same in every panel
#' 
#' Text annotations using geom_text()
#' ===
#' 
## -----------------------------------------------------------------------------
# Subset 10 rows
set.seed(32611)
ss <- sample(1:32, 10)
df <- mtcars[ss, ]

sp <- ggplot(data = df) +
  aes(wt, mpg, label = rownames(df)) +
  geom_point()
# Add texts
sp + geom_text() ## geom_text need the label aes

#'  
#' Other experiment
#' ===
#' 
#' - Change the size of the texts
#' ```{} 
#' sp + geom_text(size=6)
#' ```
#' - Change vertical and horizontal adjustement
#' ```{}
#' sp +  geom_text(hjust=0, vjust=0)
#' ```
#' - Change fontface. Allowed values : 1(normal), 2(bold), 3(italic), 4(bold.italic)
#' ```{}
#' sp + geom_text(aes(fontface=2))
#' ```
#' - Change font family
#' ```{}
#' sp + geom_text(family = "Times New Roman")
#' ```
#' - Color by groups
#' ```{}
#' sp + geom_text(aes(color=factor(cyl)))
#' ```
#' - Set the size of the text using a continuous variable
#' ```{}
#' sp + geom_text(aes(size=wt))
#' ```
#' 
#' Text annotations using geom_label()
#' ===
#' 
## -----------------------------------------------------------------------------
sp <- ggplot(data = df) +
  aes(wt, mpg, label = rownames(df)) +
  geom_point()
# Add texts
sp + geom_label()

#' 
#' 
#' Add a text annotation at a particular coordinate
#' ===
## -----------------------------------------------------------------------------
# Solution 1
sp + geom_text(x=3, y=20, label="Scatter plot")

#' 
#' 
#' ggrepel: Avoid overlapping of text labels
#' ===
#' 
#' - geom_label_repel()
#' - geom_text_repel()
#' 
## -----------------------------------------------------------------------------
library(ggrepel)

#' 
#' Create a scatter plot and add labels
#' ===
#' 
## -----------------------------------------------------------------------------
p <- ggplot(mtcars, aes(wt, mpg)) +
  geom_point(color = 'red') 
p + geom_text(aes(label = rownames(mtcars)),
              size = 3.5)

#' 
#' 
#' Use geom_text_repel
#' ===
## -----------------------------------------------------------------------------
set.seed(32611)
p + geom_text_repel(aes(label = rownames(mtcars)),
                    size = 3.5) 

#' 
#' Use label_text_repel
#' ===
## -----------------------------------------------------------------------------
set.seed(32611)
p + geom_label_repel(aes(label = rownames(mtcars)))
## p + geom_label_repel(aes(label = rownames(mtcars), fill = factor(cyl)), 
##                         color = 'white', size = 3.5
##                    )

#' 
#' Labs
#' ===
#' 
#' - title, x, y
#' 
## -----------------------------------------------------------------------------
p <- ggplot(mpg) + 
  geom_point(aes(x = displ, y = hwy, colour=factor(cyl))) + 
  labs(title = "New plot title", x = "New x label", y = "New y label")

#' 
#' 
#' Theme
#' ===
#' 
#' - theme_grey: signature ggplot2 theme.
#' - theme_bw: dark-on-light theme.
#' - theme_linedraw: flip x and y.
#' - theme_light: light grey background.
#' - theme_dark: dark background.
#' - theme_minimal: no background annotations.
#' - theme_classic: classic-looking theme.
#' - theme_void: empty theme.
#' 
#' 
#' 
#' examples on different themes
#' ===
## -----------------------------------------------------------------------------
p <- ggplot(mpg) + 
  geom_point(aes(x = displ, y = hwy, colour=factor(cyl))) + 
  facet_wrap(~class)
p ## default theme_grey

#' 
#' 
#' ---
#' 
## -----------------------------------------------------------------------------
p + theme_bw()
# p + theme_linedraw()
# p + theme_light()
# p + theme_dark()
# p + theme_minimal()
# p + theme_classic()
# p + theme_void()

#' 
#' 
#' More about the theme
#' ===
#' 
## -----------------------------------------------------------------------------
p + theme(text = element_text(size=20),
        axis.text.x = element_text(angle=90, hjust=1,colour="red")) 

#' 
#' Four elements to control the theme
#' ===
#' 
#' - element_blank(): draws nothing, and assigns no space.
#' - element_rect(): borders and backgrounds.
#' - element_line(): lines
#' - element_text(): text
#' 
#' examples
#' ===
#' 
#' ```{}
#' plot <- ggplot(mpg, aes(displ, hwy)) + geom_point()
#' 
#' plot + theme(
#'   panel.background = element_blank(),
#'   axis.text = element_blank()
#' )
#' 
#' plot + theme(
#'   axis.text = element_text(colour = "red", size = rel(1.5))
#' )
#' 
#' plot + theme(
#'   axis.line = element_line(arrow = arrow())
#' )
#' 
#' plot + theme(
#'   panel.background = element_rect(fill = "white"),
#'   plot.margin = margin(2, 2, 2, 2, "cm"),
#'   plot.background = element_rect(
#'     fill = "grey90",
#'     colour = "black",
#'     size = 1
#'   )
#' )
#' ## all changes are relative to the default value
#' ```
#' 
#' --- 
#' 
#' - Arguments in the theme function control figure theme
#' - There is a hierarchy in these arguments. (E.g., text will overwrite all text related arguments.)
#' 
#' ```{}
#' line
#' rect
#' text
#' title
#' aspect.ratio
#' axis.title
#' axis.title.x
#' axis.title.y 
#' axis.text
#' axis.text.x
#' axis.text.y
#' axis.ticks
#' axis.ticks.x
#' axis.ticks.y,
#' axis.ticks.length
#' axis.line
#' axis.line.x
#' axis.line.y
#' ## for more options, see
#' ?theme
#' ```
#' 
#' ---
#' 
#' - look at theme_gray() function
#' 
#' ```{}
#' theme_gray
#' ```
#' 
#' No legend
#' ===
## -----------------------------------------------------------------------------
ggplot(mpg) + 
  geom_point(aes(x = displ, y = hwy, colour=factor(cyl))) + 
  theme(legend.position = "none") 

#' 
#' 
#' 
#' One of my favourate themes (1)
#' ===
#' 
#' 
## -----------------------------------------------------------------------------
black.bold.text <- element_text(face = "bold", color = "black", size=20)
ggplot(mpg, aes(displ, hwy, colour=class)) + geom_point() + 
    labs(title="hwy vs displ") + 
    theme_bw() + 
    theme(text = black.bold.text) 

#' 
#' - theme() will overwrite part of theme_bw()
#' 
#' One of my favourate themes (2)
#' ===
#' 
#' 
## -----------------------------------------------------------------------------
black.bold.text <- element_text(face = "bold", color = "black", size=20)
ggplot(mpg, aes(displ, hwy, colour=class)) + geom_point() + 
    labs(title="hwy vs displ") + 
    theme_bw() + 
    theme(text = black.bold.text, panel.grid =element_blank()) 

#' 
#' Change font
#' ===
#' 
## -----------------------------------------------------------------------------
black.bold.text <- element_text(face = "bold", color = "black", size=20)
red.italic.text <- element_text(face = "italic", color = "red", size=15)

ggplot(mpg, aes(displ, hwy, colour=class)) + geom_point() + 
    labs(title="hwy vs displ") + 
    theme_bw() + 
    theme(axis.text = black.bold.text , axis.title = black.bold.text, 
          legend.title = red.italic.text, 
          legend.text = black.bold.text) 

#' 
#' Create your own discrete scale
#' ===
#' 
## -----------------------------------------------------------------------------
p <- ggplot(mtcars, aes(mpg, wt)) +
  geom_point(aes(colour = factor(cyl)))
p 

#' 
## -----------------------------------------------------------------------------
p + scale_colour_manual(values = c("red", "blue", "green"))

#' 
#' 
#' ---
#' 
#' - to use a named vector
#' 
## -----------------------------------------------------------------------------
cols <- c("8" = "red", "4" = "blue", "6" = "darkgreen", "10" = "orange")
p + scale_colour_manual(values = cols)

#' 
#' 
#' ---
#' 
#' - Set color and fill aesthetics at the same time
## -----------------------------------------------------------------------------
ggplot(mtcars) +
  aes(mpg, wt, colour = factor(cyl), fill = factor(cyl)) +
   geom_point() + 
  scale_colour_manual(
    values = cols,
    aesthetics = c("colour", "fill")
  )

#' 
#' 
#' 
#' Create your own axis ticks
#' ===
#' 
## -----------------------------------------------------------------------------
p <- ggplot(mtcars, aes(mpg, wt)) +
  geom_point(aes(colour = factor(cyl))) 
p + scale_x_continuous(breaks = c(15,25),
	labels = c("A", "B"),
	name = "My MPG") 

#' 
#' - scale_x_continuous
#' - scale_y_continuous
#' - scale_x_discrete
#' - scale_y_discrete
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
## -----------------------------------------------------------------------------
df <- data.frame(x = rnorm(1000))
ggplot(df, aes(x)) + stat_ecdf()

#' 
#' 
#' ---
#' 
## -----------------------------------------------------------------------------
n <- 100
df <- data.frame(x = c(rnorm(n, 0, 3), rnorm(n, 0, 10)),
                 g = gl(2, n))
ggplot(df, aes(x, colour = g)) + stat_ecdf()

#' 
#' stat_function
#' ===
#' 
## -----------------------------------------------------------------------------
n <- 100
set.seed(32611)
df <- data.frame(
  x = rnorm(n)
)
base <- ggplot(df, aes(x)) + geom_density()
base + stat_function(fun = dnorm, colour = "red") + xlim(c(-3,3))

#' 
#' stat_ellipse
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(mpg, aes(x = displ, y = hwy)) + geom_point() +   stat_ellipse()

#' 
#' 
#' stat_ellipse by group
#' ===
#' 
## -----------------------------------------------------------------------------
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
#' Previous example on mpg
#' ===
#' 
## -----------------------------------------------------------------------------
p <- ggplot(mpg, aes(displ, hwy)) +
  geom_point() +
  geom_smooth()

p

#' 
#' ---
#' 
#' # Setting the limits on the coordinate system performs a visual zoom.
## -----------------------------------------------------------------------------
p + coord_cartesian(xlim = c(3, 5), expand = FALSE)

#' 
#' ---
#' 
#' Setting the limits on a scale converts all values outside the range to NA.
## -----------------------------------------------------------------------------
p + xlim(3, 5)
#the same as p + scale_x_continuous(limits = c(325, 500))

#' 
#' 
#' 
#' resize the plot
#' ===
#' 
## -----------------------------------------------------------------------------
p <- ggplot(mpg, aes(displ, hwy)) +  geom_point()
p + coord_fixed(ratio = 0.5)

#' 
#' flip x and y
#' ===
#' 
## -----------------------------------------------------------------------------
ggplot(mpg, aes(class, hwy)) +
  geom_boxplot() +
  coord_flip()

#' 
#' ggplot Cheat Sheet
#' ===
#' 
#' - https://github.com/rstudio/cheatsheets/raw/master/data-visualization-2.1.pdf
#' 
