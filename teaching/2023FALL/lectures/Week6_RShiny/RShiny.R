#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Thursday September 21, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: R Shiny
#' ---
#' 
#' 
#' 
#' R Shiny
#' ===
#' 
#' - Shiny is an R package
#'   - Based on R language
#'   - Interactive web apps
#'   - Mixed with CSS themes, htmlwidgets, and JavaScript Actions.
#' 
#' - R Shiny Gallery
#' https://shiny.rstudio.com/gallery/
#'   - https://shiny.posit.co/r/gallery/start-simple/faithful/
#'   - https://shiny.posit.co/r/gallery/start-simple/kmeans-example/
#'   - https://shiny.posit.co/r/gallery/start-simple/word-cloud/
#'   - https://shiny.posit.co/r/gallery/interactive-visualizations/bus-dashboard/
#'   - https://shiny.posit.co/r/gallery/interactive-visualizations/movie-explorer/
#'   
#' 
#' 
#' Major references:
#' ===
#' 
#' Credits to these resources:
#' 
#' - https://shiny.rstudio.com/tutorial/
#' - https://github.com/rstudio-education/shiny.rstudio.com-tutorial
#' - https://shiny.rstudio.com/articles/
#' 
#' The Training Materials are licensed under the Creative Commons Attribution-Noncommercial 3.0 United States License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/3.0/us/ or send a letter to Creative Commons, 171 Second Street, Suite 300, San Francisco, California, 94105, USA.
#' 
#' 
#' Outlines
#' ===
#' 
#' 0. R Shiny gallary
#' 1. Toy examples
#' 2. Deploy R Shiny app
#' 3. Server
#' 4. UI
#' 
#' 
#' R default example
#' ===
#' 
#' ![](../figure/defaultExample.png){ width=50% }
#' 
#' We will see two key components:
#' 
#' - ui code: user interface
#' - server code: 
#'   - receive data from ui
#'   - run R code
#'   - render results in ui
#' 
#' 
#' 
#' 
#' R Shiny architecture
#' ===
#' 
#' ![](../figure/server_UI.png){ width=50% }
#' 
#' - The server:
#'   - Can be your own laptop
#'   - Or a remote server
#' 
#' 
#' A minimum R Shiny example
#' ===
#' 
#' - Minimum example
#' 
#' 
#' ```
#' library(shiny)
#' ui <- fluidPage()
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' - Add arguments to fluidPage()
#' ```
#' ui <- fluidPage("Hello World!")
#' ```
#' 
#' - Close the app
#' 
#' ![](../figure/close.png){ width=50% }
#' 
#' 
#' UI, input and output
#' ===
#' 
#' ```
#' ui <- fluidPage(
#' # *Input() functions,
#' # *Output() functions
#' )
#' ```
#' 
#' ![](../figure/input_output.png){ width=70% }
#' 
#' 
#' slider Input
#' ===
#' 
#' ```
#' # library(shiny)
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",
#'               label = "Choose a number",
#'               value = 25, min = 1, max = 100)
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' ![](../figure/sliderInput.png){ width=50% }
#' 
#' - inputId: to interact with the server
#' - label: to appear on the slider
#' - value = 25, min = 1, max = 100: default value and range of the slider
#' 
#' More on input
#' ===
#' ![](../figure/input.png){ width=80% }
#' 
#' ---
#' 
#' Try each one of these components
#' 
#' ```
#' # library(shiny)
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",label = "Choose a number",value = 25, min = 1, max = 100),
#'   actionButton(inputId = "actionButton",label="Action"),
#'   submitButton(text = "Submit"),
#'   checkboxInput(inputId = "checkbox", label = "Choice A"),
#'   checkboxGroupInput(inputId = "checkboxgroup", label = "Checkbox group", choices = c("A"="Choice 1", "B"="Choice 2")),
#'   dateInput(inputId = "dateInput", label = "date input", value = "2021-09-20"),
#'   fileInput(inputId = "fileInput", label = "File input"),
#'   numericInput(inputId = "numericInput", label = "Numeric input", value = "10"),
#'   textInput(inputId = "textInput", label = "Text input", value = "Enter text..."),
#'   radioButtons(inputId = "radioButton", label = "Radio buttons", choices = c("A"="Choice 1", "B"="Choice 2")),
#'   selectInput(inputId = "selectBox", label = "Select box", choices = c("A"="Choice 1", "B"="Choice 2"))
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' Also see: https://shiny.rstudio.com/gallery/widget-gallery.html
#' 
#' Output
#' ===
#' 
#' To display output, add an **Output()** function to the fluidPage(), separated by ","
#' 
#' ```
#' # library(shiny)
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",
#'               label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   plotOutput(outputId="hist")
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' In this example,
#' 
#' - plotOutput: specifies the type of output to display in UI
#' - "hist": name to give to the output object
#' - separate input and output by ","
#' - build the output object in the server function
#' 
#' 
#' More on output
#' ===
#' 
#' ![](../figure/output.png){ width=80% }
#' 
#' 
#' 
#' A simple server response
#' ===
#' 
#' ```
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   plotOutput(outputId="hist")
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'     hist(rnorm(100))
#'   })}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' - output argument in server matches outputID in ui
#'   - hist appears in both ui and server
#' - Build objects to display with render*()
#'   
#' 
#' ![](../figure/simple1.png){ width=50% }
#' 
#' 
#' 3 rules for the server function
#' ===
#' 
#' - save objects to display to output$
#' - build objects to display with render*()
#' - access input values with input$
#' 
#' Example 0
#' 
#' ```
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   plotOutput(outputId="hist")
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'     hist(rnorm(input$num))
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' 
#' Input values
#' ===
#' 
#' input argument in server matches inputID in ui
#' 
#' ![](../figure/inputValues.png){ width=80% }
#' 
#' Output will automatically update if you follow the 3 rules
#' 
#' 
#' More on render
#' ===
#' 
#' ![](../figure/render.png){ width=80% }
#' 
#' 
#' Another example on table render
#' ===
#' 
#' ```
#' library(shiny)
#' ui <- fluidPage(
#'     selectInput(inputId = "selectBox", label = "Select Species", choices = c("setosa"="setosa", "versicolor"="versicolor", "virginica"="virginica")),
#'     dataTableOutput(outputId="table")
#' )
#' 
#' server <- function(input, output) {
#'     output$table <- renderDataTable({
#'         iris[iris$Species==input$selectBox,]
#'     })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' 
#' 
#' Run Shiny apps
#' ===
#' 
#' Every Shiny app is maintained by a computer running R
#' 
#' - local laptop
#' ![](../figure/R_laptop.png){ width=70% }
#' 
#' - remote server
#' ![](../figure/R_server.png){ width=70% }
#' 
#' 
#' Run Shiny apps on local laptop
#' ===
#' 
#' - Click on Run App
#' 
#' ![](../figure/RunApp.png){ width=70% }
#' 
#' - runApp 
#'   - save the code as app.R
#' 
#' ```
#' setwd("/Users/zhuo/Desktop") ## change to your path
#' runApp("test", display.mode = "showcase")
#' ```
#' 
#' runApp
#' ===
#' 
#' save the code as ui.R and server.R
#' 
#' ![](../figure/twoFilesApp.png){ width=70% }
#' 
#' 
#' ```
#' setwd("/Users/zhuo/Desktop") ## change to your path
#' runApp("test", display.mode = "showcase")
#' ```
#' 
#' mode: 
#' 
#' - auto
#' - normal
#' - showcase
#' 
#' 
#' Run Shiny apps on server
#' ===
#' 
#' There are many servers.
#' - Shinyapps.io is a server maintained by R studio.
#'   - Free for basic usage.
#' 
#' 
#' ![](../figure/shinyapps.png){ width=70% }
#' 
#' 
#' Deploy on Shinyapps.io
#' ===
#' 
#' Reference: https://shiny.rstudio.com/articles/shinyapps.html
#' 
#' 1. Go to https://www.shinyapps.io/
#' 
#' 2. Sign in and setup your account
#' 
#' ![](../figure/setup_URL.png){ width=100% }
#' 
#' 
#' 
#' 3. Follow the instruction
#' - INSTALL RSCONNECT
#' ```
#' install.packages('rsconnect')
#' ```
#' 
#' - AUTHORIZE ACCOUNT (need to click show secret)
#' ```
#' rsconnect::setAccountInfo
#' ```
#' 
#' - DEPLOY
#' ```
#' rsconnect::deployApp('path/to/your/app') ## setup account if you have multiple accounts
#' ```
#' 
#' Server response: Reactivity
#' ===
#' 
#' ![](../figure/inputValues.png){ width=70% }
#' 
#' 
#' ```
#' server <- function(input, output) {
#'   output$hist <- renderPlot({ hist(rnorm(100, input$num)) })
#' }
#' ```
#' 
#' - Reactive value: input$num
#'   - The input value changes whenever a user changes the input.
#' - Reactive function: renderPlot
#'   - Reactive values work together with reactive functions.
#'   - in Rshiny, you cannot use a reactive value in ui without calling a reactive function.
#' 
#' ```
#' server <- function(input, output) {
#'   output$hist <- hist(rnorm(input$num)) ## this won't work
#' }
#' ```
#' 
#' 
#' Reactivity in R is a two-step proces
#' ===
#' 
#' 1. Reactive values notify the Reactive functions when they become invalid. This happens when you update your input values. 
#'   
#' 2. The Reactive functions refresh. This will update the output result. 
#' 
#' Reactive functions:
#' 
#' ![](../figure/render.png){ width=80% }
#' 
#' 
#' Example 1, two inputs
#' ===
#' 
#' ```
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",
#'               label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   textInput(inputId = "title",
#'             label = "Write a title",
#'             value = "Histogram of Random Normal Values"),
#'   plotOutput("hist")
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'           hist(rnorm(input$num), main = input$title)
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' When notified that the reactive function is invalid, the object created by a render*() function will rerun the entire block of code associated with it.
#' 
#' Example 2, two outputs
#' ===
#' 
#' ```
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",
#'               label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   plotOutput("hist"),
#'   verbatimTextOutput("stats")
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'     hist(rnorm(input$num))
#'   })
#'   output$stats <- renderPrint({
#'     summary(rnorm(input$num))
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' Problem: input$num for hist and stats are not the same.
#' 
#' Example 2, two outputs
#' ===
#' 
#' reactive() function builds a reactive object
#' - will make sure the same input for all downstream functions
#' 
#' 
#' ```
#' data <- reactive( {rnorm(input$num)})
#' ```
#' 
#' ![](../figure/twooutputs_same.png){ width=70% }
#' 
#' Example 3, reactive
#' ===
#' 
#' ```
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",
#'               label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   plotOutput("hist"),
#'   verbatimTextOutput("stats")
#' )
#' server <- function(input, output) {
#'   data <- reactive({
#'     rnorm(input$num)
#'   })
#'   output$hist <- renderPlot({
#'     hist(data())
#'   })
#'   output$stats <- renderPrint({
#'     summary(data())
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' - reactive() makes an object to use in the downstream functions
#' - modulize input object
#' - it caches input values to avoid unnecessary computation
#' 
#' Prevent auto update -- isolate()
#' ===
#' 
#' - isolate()
#'   - Object will NOT respond to any reactive value in the code
#'   - makes an non-reactive object
#' 
#' ```
#' isolate({rnorm(input$num)})
#' ```
#' 
#' 
#' 
#' - Example 4, isolate
#' 
#' ```
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",
#'               label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   textInput(inputId = "title",
#'             label = "Write a title",
#'             value = "Histogram of Random Normal Values"),
#'   plotOutput("hist")
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'     hist(rnorm(input$num),
#'          main = isolate({input$title}))
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' Trigger code -- observeEvent()
#' ===
#' 
#' - ui:
#'   - actionButton()
#' - server:
#'   - observeEvent()
#' 
#' Example 5, actionButton
#' 
#' ```
#' library(shiny)
#' ui <- fluidPage(
#'   actionButton(inputId = "clicks",
#'                label = "Click me")
#' )
#' server <- function(input, output) {
#'   observeEvent(input$clicks, {
#'     print(as.numeric(input$clicks))
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' - more about action buttons:
#' https://shiny.rstudio.com/articles/action-buttons.html
#' 
#' 
#' 
#' Delay reactions -- eventReactive()
#' ===
#' 
#' Example 6, eventReactive()
#' 
#' ```
#' ui <- fluidPage(
#'   sliderInput(inputId = "num",
#'               label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   actionButton(inputId = "go",
#'                label = "Update"),
#'   plotOutput("hist")
#' )
#' server <- function(input, output) {
#'   data <- eventReactive(input$go, {
#'     rnorm(input$num)
#'   })
#'   output$hist <- renderPlot({
#'     hist(data())
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' compare eventReactive with observeEvent
#' ===
#' 
#' They are identical but for one thing:
#' 
#' - observeEvent() just executes code within it's boundaries but does not assign the output to anything
#' - eventReactive() will save the output to a reactive variable
#' 
#' 
#' 
#' Manage state -- reactiveValues()
#' ===
#' 
#' - reactiveValues()
#'   - create your own reactive value in server
#'   - you can manipulate these values, usually with observeEvent()
#' 
#' Example 7 -- reactiveValues()
#' ```
#' ui <- fluidPage(
#'   actionButton(inputId = "norm", label = "Normal"),
#'   actionButton(inputId = "unif", label = "Uniform"),
#'   plotOutput("hist")
#' )
#' server <- function(input, output) {
#'   rv <- reactiveValues()
#'   rv$data <- rnorm(100)
#'   observeEvent(input$norm, { rv$data <- rnorm(100) })
#'   observeEvent(input$unif, { rv$data <- runif(100) })
#'   output$hist <- renderPlot({
#'     hist(rv$data)
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' Summarize
#' ===
#' 
#' 
#' ![](../figure/summary.png){ width=80% }
#' 
#' - render reactive output: render*()
#' - modularize reactions: reactive()
#' - prevent reactions: isolate()
#' - trigger arbitrary code: observeEvent(), observe()
#' - delay reactions: eventReactive()
#' - create your own reactive values: reactiveValues()
#' 
#' 
#' More on UI (user interface)
#' ===
#' 
#' ![](../figure/UI.png){ width=80% }
#' 
#' 
#' The R Shiny UI was built by HTML, try the following in R console:
#' 
#' 
## ----------------------------------------------------------------
library(shiny)

#' 
#' 
#' ```
#' ui <- fluidPage()
#' ui
#' ```
#' 
#' ```
#' sliderInput(inputId = "num",
#'   label = "Choose a number",
#'   value = 25, min = 1, max = 100)
#' ```
#' 
#' ```
#' plotOutput("hist")
#' ```
#' 
#' Basic html code
#' ===
#' 
#' ```
#' <div class="container-fluid">
#'   <h1>My Shiny App</h1>
#'   <p style="font-family:Impact">
#'     See other apps in the
#'     <a href="http://www.rstudio.com/products/shiny/shiny-usershowcase/">
#'     Shiny Showcase</a>
#'   </p>
#' </div>
#' ```
#' 
#' Save it to a html file, and open it.
#' 
#' When writing HTML, add content with tags:
#' 
#' ```
#' <h1></h1>
#' <a></a>
#' ```
#' 
#' How to add content to a web page in R Shiny
#' ===
#' 
#' When writing R, add content with tag functions
#' 
#' 
#' ![](../figure/tags.png){ width=40% }
#' 
#' 
## ----------------------------------------------------------------
names(tags)

#' 
#' 
#' tags is a list of functions
#' ===
#' 
#' ```
#' tags$h1
#' ```
#' 
#' ```
#' tags$h1()
#' ```
#' 
#' ```
#' tags$h1("this is a header")
#' ```
#' 
#' ```
#' tags$a(href = "www.rstudio.com", "RStudio")
#' ```
#' 
#' ```
#' ui <- fluidPage(
#'   tags$h1("this is a header"), 
#'   tags$a(href = "www.rstudio.com", "RStudio")
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' h1() - h6()
#' ===
#' 
#' ```
#' ui <- fluidPage(
#'   tags$h1("First level"),
#'   tags$h2("Second level"),
#'   tags$h3("Third level"),
#'   tags$h4("Fourth level"),
#'   tags$h5("Fifth level"),
#'   tags$h6("Sixth level")
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' text and paragraph
#' ===
#' 
#' - text
#' 
#' ```
#' ui <- fluidPage(
#'  "This is a Shiny app.",
#'   "It is also a web page."
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' 
#' - paragraph
#' 
#' ```
#' ui <- fluidPage(
#'   tags$p("This is a Shiny app."),
#'   tags$p("It is also a web page.")
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' styles
#' ===
#' 
#' - emphasized (italic): em()
#' - strong (bold): strong()
#' - code: code()
#' - next functions inside others
#' - line break: br()
#' - horizontal rule: hr()
#' - image: img() with src argument
#' 
#' ```
#' ui <- fluidPage(
#'  tags$em("This is a Shiny app."), 
#'  tags$br(),
#'  tags$strong("This is a Shiny app."),
#'  tags$hr(),
#'  tags$code("This is a Shiny app."), 
#'  tags$p("This is a", tags$strong("Shiny"), "app."),
#'  tags$img(height=100, width=100, src="https://caleb-huo.github.io/teaching/2023FALL/logo.png")
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' Adding images from local file
#' ===
#' 
#' To add images from local files, save the file in a subdirectory named - <span style="color:blue">www</span>
#' 
#' 
#' ![](../figure/www.png){ width=80% }
#' 
#' Summary of common tags
#' ===
#' 
#' ![](../figure/tags2.png){ width=80% }
#' 
#' 
#' Raw HTML code
#' ===
#' 
#' Use HTML() to pass a character string as raw HTML
#' 
#' ```
#' ui <- fluidPage(
#'  HTML("<h1>My Shiny App</h1>")
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' More reference on HTML UI:
#' 
#' shiny.rstudio.com/articles/html-ui.html
#' 
#' 
#' layout functions
#' ===
#' 
#' - fluidRow()
#'   - add a new row.
#'   - each new row goes below the previous rows.
#' 
#' ![](../figure/rows.png){ width=50% }
#'   
#' - column(width = 2)
#'   - adds columns within a row. 
#'   - each new column goes to the left of the previous column.
#'   - Specify the width and offset of each column out of 12
#' 
#' ![](../figure/fluid12.png){ width=60% }
#' 
#' 
#' layout functions examples
#' ===
#' 
#' ```
#' ui <- fluidPage(
#'   fluidRow(
#'     column(3),
#'     column(5, 
#'            sliderInput(inputId = "num",label = "Choose a number",
#'                        value = 25, min = 1, max = 100)
#'     )
#'   ),
#'   fluidRow(
#'     column(4, offset = 8,
#'            plotOutput(outputId="hist")
#'     )
#'   )
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'     hist(rnorm(input$num))
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' ![](../figure/columns.png){ width=50% }
#' 
#' 
#' 
#' Panels
#' ===
#' 
#' Panels to group multiple elements into a single unit with its own properties.
#' 
#' - wellPanel()
#'   - Groups elements into a grey "well"
#'   
#' 
#' ```
#' ui <- fluidPage(
#'   #wellPanel(
#'   sliderInput(inputId = "num",label = "Choose a number",
#'               value = 25, min = 1, max = 100),
#'   textInput("title", value = "Histogram", label = "Write a title"),
#'   #),
#'   plotOutput(outputId="hist")
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'           hist(rnorm(input$num), main = input$title)
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' More on panels
#' ===
#' 
#' ![](../figure/panels.png){ width=90% }
#' 
#' 
#' tabsetPanel()
#' ===
#' 
#' - tabsetPanel() 
#'   - combines tabs into a single panel.
#'   - Use tabs to navigate between tabs.
#'   
#' ```
#' ui <- fluidPage(
#'   tabsetPanel(
#'     tabPanel("tab 1", "contents"),
#'     tabPanel("tab 2", "contents"),
#'     tabPanel("tab 3", "contents")
#'     )
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' navlistPanel()
#' ===
#' 
#' - navlistPanel() 
#'   - combines tabs into a single panel.
#'   - Use links to navigate between tabs
#' 
#' 
#' ```
#' ui <- fluidPage(
#'   navlistPanel(
#'     tabPanel("tab 1", "contents"),
#'     tabPanel("tab 2", "contents"),
#'     tabPanel("tab 3", "contents")
#'     )
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' More reference about layout: https://shiny.rstudio.com/articles/layout-guide.html
#' 
#' Prepackaged layout -- sidebarLayout()
#' ===
#' 
#' ```
#' ui <- fluidPage(
#'   sidebarLayout(
#'     sidebarPanel(),
#'     mainPanel()
#'   )
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' ---
#' 
#' ```
#' ui <- fluidPage(
#'   sidebarLayout(
#'     sidebarPanel(
#'       sliderInput(inputId = "num",label = "Choose a number",
#'                   value = 25, min = 1, max = 100),
#'       textInput("title", value = "Histogram", label = "Write a title")
#'     ),
#'     mainPanel(
#'       plotOutput(outputId="hist")
#'     )
#'   )
#' )
#' server <- function(input, output) {
#'   output$hist <- renderPlot({
#'     hist(rnorm(input$num), main = input$title)
#'   })
#' }
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' Prepackaged layout -- navbarPage()
#' ===
#' 
#' - navbarPage() combines tabs into a single page.
#' - navbarPage() replaces fluidPage(). 
#' - Requires title.
#' 
#' ```
#' ui <- navbarPage( title = "Title",
#'             tabPanel("tab 1", "contents"),
#'             tabPanel("tab 2", "contents"),
#'             tabPanel("tab 3", "contents")
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' 
#' Prepackaged layout -- navbarMenu()
#' ===
#' 
#' - navbarMenu() combines tab links into a dropdown menu for navbarPage()
#' 
#' ```
#' ui <- navbarPage(title = "Title",
#'           tabPanel("tab 1", "contents"),
#'           tabPanel("tab 2", "contents"),
#'           navbarMenu(title = "More",
#'                 tabPanel("tab 3", "contents"),
#'                 tabPanel("tab 4", "contents"),
#'                 tabPanel("tab 5", "contents")
#'           )
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' Prepackaged layout -- dashboardPage()
#' ===
#' 
#' dashboardPage() comes in the shinydashboard package
#' 
#' ```
#' library(shinydashboard)
#' ui <- dashboardPage(
#'           dashboardHeader(),
#'           dashboardSidebar(),
#'           dashboardBody()
#' )
#' server <- function(input, output) {}
#' shinyApp(ui = ui, server = server)
#' ```
#' 
#' 
