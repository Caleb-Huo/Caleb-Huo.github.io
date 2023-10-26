#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Thursday Oct 26th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Shiny for Python"
#' ---
#' 

#' 
#' Outline
#' ===
#' 
#' - A minimum examples
#' - How to deploy the Python Shiny app
#' - Basics on UI and Server
#' - More on this topic next week...
#' 
#' 
#' Python Shiny application gallery
#' ===
#' 
#' https://shiny.posit.co/py/gallery/
#' 
#' 
#' 
#' A minimum examples
#' ===
#' 
#' reference: https://shiny.posit.co/py/docs/install.html
#' 
#' 1. Open Anaconda, 
#'     - create/select an environment
#'         - recommend a new environment
#'     - create a folder and go into the folder
#' 
#' 2. In Python terminal, install the required packages
#' 
#' ```
#' pip install --upgrade pip wheel
#' pip install shiny
#' pip install --upgrade shiny htmltools
#' ```
#' 
#' 3. In Python terminal, create a Python Shiny using the default
#' 
#' ```
#' cd your_working_directory ## go to your working directory
#' shiny create . ## This will create a basic Shiny application in the current directory, in a file named app.py.
#' shiny run --reload
#' ```
#' 
#' 4. If the web application doesn't show up, copy the URL into your browser (e.g., http://127.0.0.1:8000/)
#' 
#' 
#' 
#' Deploy on shinyapps.io
#' ===
#' 
#' follow the instruction on https://www.shinyapps.io/admin/#/dashboard
#' 
#' 1. INSTALL RSCONNECT-PYTHON
#' ```
#' pip install rsconnect-python
#' ```
#' 
#' 2. AUTHORIZE ACCOUNT
#' 
#' - replace biostatisticscomputing with your account
#' - replace token with your token
#' - replace <SECRET> with your secret
#' 
#' ```
#' rsconnect add \
#' 	  --account biostatisticscomputing \
#' 	  --name biostatisticscomputing \
#' 	  --token 9C753A57C8E5CE5E00F92F109C0B172B \
#' 	  --secret <SECRET>
#' ```
#' 
#' 
#' 3. DEPLOY
#' 
#' ```
#' rsconnect deploy shiny path/to/your/app --name biostatisticscomputing --title your-app-name
#' ```
#' 
#' 
#' A minimum example
#' ===
#' 
#' ```
#' from shiny import App, ui
#' 
#' # Part 1: ui ----
#' app_ui = ui.page_fluid(
#'     "Hello, world!",
#' )
#' 
#' # Part 2: server ----
#' def server(input, output, session):
#'     ...
#' 
#' # Combine into a shiny app.
#' # Note that the variable must be "app".
#' app = App(app_ui, server)
#' 
#' ```
#' 
#' Adding UI inputs and outputs
#' ===
#' 
#' ```
#' from shiny import App, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_slider("n", "Choose a number n:", 0, 100, 40),
#'     ui.output_text_verbatim("txt")
#' )
#' 
#' def server(input, output, session):
#'     pass
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' Adding server logic
#' ===
#' 
#' - define a function named txt, whose output shows up in the UI’s output_text_verbatim("txt").
#' - decorate it with @render.text, to say the result is text (and not, e.g., an image).
#' - decorate it with @output, to say the result should be displayed on the web page.
#' 
#' ```
#' from shiny import ui, render, App
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_slider("n", "N", 0, 100, 40),
#'     ui.output_text_verbatim("txt"),
#' )
#' 
#' def server(input, output, session):
#'     @output
#'     @render.text
#'     def txt():
#'         return f"n*2 is {input.n() * 2}"
#' 
#' # This is a shiny.App object. It must be named `app`.
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' Reactive flow
#' ===
#' 
#' ![](../figure/reactive.png){width=100%}
#' 
#' 
#' More on this topic next week
#' ===
#' 
#' Will follow close to this tutorial:
#' 
#' - https://shiny.posit.co/py/docs/overview.html
#' 
#' 
#' References:
#' ===
#' 
#' - https://shiny.posit.co/py/docs/overview.html
#' - https://shiny.posit.co/py/docs/install.html
#' 
