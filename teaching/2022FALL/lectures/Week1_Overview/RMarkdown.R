#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday August 24, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: R markdown
#' ---
#' 
#' 
#' 
#' R markdown
#' ===
#' 
#' - R Markdown provides an authoring framework for data science. You can use a single R Markdown file to:
#'    - save and execute code
#'    - generate high quality reports that can be shared with your advisor
#'    - reproduce your result
#'    - Homeworks/exam/final project should be finished using R Markdown.
#' 
#' - R markdown cheatsheet: [https://www.rstudio.com/wp-content/uploads/2015/02/rmarkdown-cheatsheet.pdf](https://www.rstudio.com/wp-content/uploads/2015/02/rmarkdown-cheatsheet.pdf)
#' 
#' - R Markdown Reference Guide:
#' https://www.rstudio.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf
#' 
#' Demonstrate R Markdown in R studio
#' ===
#' 
#' - create R markdown file
#' - compile the example file
#' 
#' Work flow:
#' 
#' - Open - Open a file that uses the .Rmd extension.
#' - Write - Write content with the easy to use R Markdown syntax.
#' - Embed - Embed R code that creates output to include in the report.
#' - Render - Replace R code with its output and transform the report into a slideshow, pdf, html or ms Word file.
#' 
#' 
#' The R markdown default example
#' ===
#' 
#' Demonstrate how it works
#' 
#' 
#' Open File
#' ===
#' 
#' Start by saving a text file with the extension .Rmd, or open an RStudio Rmd template
#' 
#' 
#' Markdown Next
#' ===
#' 
#' - Plain text
#' - End a line with two spaces to start a new paragraph.
#' - *italics* and _italics_
#' - **bold** and __bold__
#' - <span style="color:red">color</span>
#' - superscript^2^
#' - ~~strikethrough~~
#' - [link](www.rstudio.com)
#' 
#' ***
#' 
#' - # Header 1
#' - ## Header 2
#' - ### Header 3
#' - #### Header 4
#' - ##### Header 5
#' - ###### Header 6
#' 
#' ***
#' 
#' - endash: --
#' - emdash: ---
#' - ellipsis: ...
#' - inline equation: $A = \pi \times r^{2}$, following latex format [http://www.codecogs.com/latex/eqneditor.php](http://www.codecogs.com/latex/eqneditor.php)
#' - horizontal rule (or slide break): ***
#' 
#' ***
#' 
#' - How io import an external image: 
#' 
#' ![](https://caleb-huo.github.io/teaching/2022FALL/logo.png) 
#' 
#' 
#' Unordered list
#' ===
#' 
#' * unordered list
#' * item 2
#'     + sub-item 1
#'     + sub-item 2
#' 
#' 
#' Ordered list
#' ===
#' 
#' 1. ordered list
#' 2. item 2
#'     + sub-item 1
#'     + sub-item 2
#'     
#' 
#' Markdown Table
#' ===
#' 
#' Table Header  | Second Header
#' ------------- | -------------
#' Table Cell    | Cell 2
#' Cell 3        | Cell 4
#' 
#' 
#' Embed Code 
#' ===
#' 
#' - Two plus two equals `r 2 + 2`
#' - Here’s some code, show R code and evaluate
## -----------------------------------------------------------------------------
dim(iris)

#' - Here’s some code, show R code but not evaluate
## ----eval=FALSE---------------------------------------------------------------
## dim(iris)

#' 
#' - Here’s some code, do not show R code but evaluate
## ----echo=FALSE---------------------------------------------------------------
dim(iris)

#' 
#' 
#' 
#' Rmd figure
#' ===
#' 
## ---- echo=FALSE--------------------------------------------------------------
plot(pressure)

#' 
#' Rmd table
#' ===
#' 
## ----echo = FALSE-------------------------------------------------------------
suppressWarnings(library(knitr))
kable(mtcars[1:5, ], caption = "A knitr kable.")

#' 
#' Several packages support making beautiful tables with R, such as
#' 
#' * [xtable](https://cran.r-project.org/web/packages/xtable/)
#' * [stargazer](https://cran.r-project.org/web/packages/stargazer/)
#' * [pander](http://rapporter.github.io/pander/)
#' * [tables](https://cran.r-project.org/web/packages/tables/)
#' * [ascii](http://eusebe.github.io/ascii/)
#' * etc.
#' 
#' 
#' How to extract R code from Rmd files
#' === 
#' 
## -----------------------------------------------------------------------------
suppressWarnings(library(knitr))
purl("RMarkdown.rmd", output = "RMarkdown.R ", documentation = 2)

#' 
#' 
#' Render
#' === 
#' Click the knit HTML button at the top of the RStudio scripts pane
#' 
#' - execute each embedded code chunk and insert the results into your report
#' - build a new version of your report in the output file type
#' - open a preview of the output file in the viewer pane
#' - save the output file in your working directory 
#'  
#' 
#' Other output options
#' ===
#' output:
#' 
#' - pdf_document
#' - html_document
#' - word_document
#' - ioslides_presentation
#' - slidy_presentation
#' - beamer_presentation
#' 
#' 
#' 
#' Try your self
#' ===
#' 
#' - ordered list and unordered list
#' - colored text, bold text
#' - import a figure
#' - embeded R code, R code chunk
#' - Rmd figure, Rmd tables
#' - an equation
#' - html report
#' - pdf report
#' - make a slide 
#' 
