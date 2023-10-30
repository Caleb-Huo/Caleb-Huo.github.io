#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Tue Oct 31st, 2023"
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
#' - Input controls
#' - Output controls
#' - Server logic (Reactive programming)
#' - Page layout
#' 
#' 
#' 
#' Review how to deploy
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
#' Input controls
#' ===
#' 
#' - Number inputs
#' - Text inputs
#' - Selection inputs
#' - Toggle inputs
#' - Date inputs
#' - Action inputs
#' 
#' 
#' Number inputs
#' ===
#' 
#' ```
#' from shiny import ui, App
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_numeric("x1", "Number", value=10),
#'     ui.input_slider("x2", "Slider", value=10, min=0, max=20),
#'     ui.input_slider("x3", "Range slider", value=(6, 14), min=0, max=20)
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' ![](../figure/input_number.png){width=50%}
#' 
#' 
#' Text inputs
#' ===
#' 
#' ```
#' from shiny import ui, App
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_text("x1", "Text", placeholder="Enter text"),
#'     ui.input_text_area("x2", "Text area", placeholder="Enter text"),
#'     ui.input_password ("x3", "Password", placeholder="Enter password"),
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' ![](../figure/input_text.png){width=50%}
#' 
#' 
#' Selection inputs
#' ===
#' 
#' ```
#' from shiny import ui, App
#' 
#' choices = ["apple", "banana", "cherry", "orange", "pear"]
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_selectize("x1", "Selectize (single)", choices),
#'     ui.input_selectize("x2", "Selectize (multiple)", choices, multiple = True),
#'     ui.input_select("x3", "Select (single)", choices),
#'     ui.input_select("x4", "Select (multiple)", choices, multiple = True),
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' ![](../figure/input_select.png){width=50%}
#' 
#' 
#' 
#' Selection inputs (part 2)
#' ===
#' 
#' ```
#' from shiny import ui, App
#' 
#' choices = {"a": "Choice A", "b": "Choice B", "c": "Choice C"}
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_radio_buttons("x1", "Radio buttons", choices),
#'     ui.input_checkbox_group("x2", "Checkbox group", choices),
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' ![](../figure/input_select2.png){width=30%}
#' 
#' 
#' Toggle inputs
#' ===
#' 
#' ```
#' from shiny import ui, App
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_checkbox("x1", "Checkbox"),
#'     ui.input_switch("x2", "Switch")
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' ![](../figure/input_toggle.png){width=30%}
#' 
#' 
#' Date inputs
#' ===
#' 
#' 
#' ```
#' from shiny import ui, App
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_date("x1", "Date input"),
#'     ui.input_date_range("x2", "Date range input"),
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' ![](../figure/input_date.png){width=50%}
#' 
#' 
#' Action input
#' ===
#' 
#' 
#' ```
#' from shiny import ui, App
#' 
#' app_ui = ui.page_fluid(
#'     ui.p(ui.input_action_button("x1", "Action button")),
#'     ui.p(ui.input_action_button("x2", "Action button", class_="btn-primary")),
#'     ui.p(ui.input_action_link("x3", "Action link")),
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' ![](../figure/input_action.png){width=50%}
#' 
#' 
#' 
#' Ouotput controls
#' ===
#' 
#' 
#' - Text output
#' - Table output
#' - Plot output
#' 
#' 
#' Text output
#' ===
#' 
#' - decorate it with @render.text, to say the result is text (and not, e.g., an image).
#' - decorate it with @output, to say the result should be displayed on the web page.
#' - You could define multiple functions in the server.
#' 
#' ```
#' from shiny import ui, render, App
#' 
#' app_ui = ui.page_fluid(
#'     ui.output_text("txt"),
#'     ui.output_text_verbatim("txt2"),
#' )
#' 
#' def server(input, output, session):
#'     @output
#'     @render.text
#'     def txt():
#'         return "text part 1"
#' 
#'     @output
#'     @render.text
#'     def txt2():
#'         return "text part 2"
#' 
#' # This is a shiny.App object. It must be named `app`.
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' 
#' Table output
#' ===
#' 
#' - Render various kinds of data frames into an HTML table with ui.output_table() and @render.table.
#' 
import seaborn as sns
tips = sns.load_dataset("tips")
tips.head()
#' 
#' ```
#' import pandas as pd
#' from shiny import ui, render, App
#' import seaborn as sns
#' tips = sns.load_dataset("tips")
#' 
#' app_ui = ui.page_fluid(
#'     ui.output_table("show_head"),
#' )
#' 
#' def server(input, output, session):
#'     @output
#'     @render.table
#'     def show_head():
#'         return tips[:10]
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' Static plot output
#' ===
#' 
#' - Render plots with ui.output_plot() and @render.plot
#' - Need to install the package before using them
#'   - matplotlib (this page)
#'   - seaborn (next page)
#' 
#' ```
#' from shiny import ui, render, App
#' from matplotlib import pyplot as plt
#' 
#' app_ui = ui.page_fluid(
#'     ui.output_plot("plot_matplotlib"),
#' )
#' 
#' def server(input, output, session):
#'     @output
#'     @render.plot
#'     def plot_matplotlib():
#'         return plt.scatter([1,2,3], [5, 2, 3])
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' ---
#' 
#' 
#' 
#' ```
#' from shiny import ui, render, App
#' import seaborn as sns
#' tips = sns.load_dataset("tips")
#' 
#' app_ui = ui.page_fluid(
#'     ui.output_plot("scatter_seaborn"),
#' )
#' 
#' def server(input, output, session):
#'     @output
#'     @render.plot
#'     def scatter_seaborn():
#'         return sns.relplot(data=tips,x="total_bill", y="tip", kind='scatter')
#' 
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' Reactive programming
#' ===
#' 
#' - reactive.Value: An object that causes reactive functions to run when its value changes.
#' - reactive.Calc: Create a reactive function that is used for its return value.
#' - reactive.Effect: Create a reactive function that is used for its side effects (and not its return value).
#' - @output: outputs (wrapped up in a reactive.Effect)
#' There are a few utility functions that help with managing reactivity:
#' 
#' - with isolate(): Run blocks of code inside a reactive function, but without taking a reactive dependency on code inside the block.
#' - @reactive.event(): A decorator that controls what causes a reactive function to invalidate.
#' 
#' 
#' Reactive calculations
#' ===
#' 
#' Sometimes we want to compute values based on inputs to avoid randomness
#' 
#' ```
#' from shiny import ui, render, App
#' import random
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_slider("n", "N", 0, 100, 40),
#'     ui.output_text_verbatim("txt1"),
#'     ui.output_text_verbatim("txt2"),
#' )
#' 
#' def server(input, output, session):
#'     @output
#'     @render.text
#'     def txt1():
#'         x1 = [ random.random() for i in range(input.n())]
#'         return f"The mean of {input.n()} random values is {sum(x1)/len(x1)}"
#' 
#'     @output
#'     @render.text
#'     def txt2():
#'         x2 = [ random.random() for i in range(input.n())]
#'         return f"The mean of {input.n()} random values is {sum(x2)/len(x2)}"
#' 
#' 
#' # This is a shiny.App object. It must be named `app`.
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' 
#' Reactive calculations
#' ===
#' 
#' - need to import reactive from shiny package
#' - @reactive.Calc will cache the intermediate result and guarantee consistency
#' - try a function without @reactive.Calc
#' 
#' ```
#' from shiny import ui, render, reactive, App
#' import random
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_slider("n", "N", 0, 100, 40),
#'     ui.output_text_verbatim("txt1"),
#'     ui.output_text_verbatim("txt2"),
#' )
#' 
#' 
#' 
#' def server(input, output, session):
#'     @reactive.Calc
#'     def get_mean():
#'         x = [ random.random() for i in range(input.n())]
#'         return sum(x)/len(x)
#' 
#'     @output
#'     @render.text
#'     def txt1():
#'         return f"The mean of {input.n()} random values is {get_mean()}"
#' 
#'     @output
#'     @render.text
#'     def txt2():
#'         return f"The mean of {input.n()} random values is {get_mean()}"
#' 
#' 
#' # This is a shiny.App object. It must be named `app`.
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' 
#' Handling events
#' ===
#' 
#' - prevent auto update: reactive.event()
#'     - This decorator takes one or more reactive dependency that cause the decorated function to re-execute.
#' 
#' ```
#' from shiny import App, reactive, render, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_slider("n", "N", min=1, max=100, value=1),
#'     ui.input_action_button("compute", "Compute!"),
#'     ui.output_text_verbatim("result", placeholder=True),
#' )
#' 
#' def server(input, output, session):
#' 
#'     @output
#'     @render.text
#'     @reactive.event(input.compute) # Take a dependency on the button
#'     def result():
#'         # Because of the @reactive.event(), everything in this function is
#'         # ignored until reactive dependencies are triggered.
#'         return f"Result: {input.n()}"
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' Handling events
#' ===
#' 
#' - prevent auto update: reactive.isolate()
#'     - Using with isolate(), a block of code won't run until a reactive dependency is triggered.
#'     - the reactive dependency should be in the same function of the  reactive.isolate()
#' 
#' ```
#' from shiny import App, reactive, render, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_slider("n", "N", min=1, max=100, value=1),
#'     ui.input_action_button("compute", "Compute!"),
#'     ui.output_text_verbatim("result", placeholder=True),
#' )
#' 
#' def server(input, output, session):
#' 
#'     @output
#'     @render.text
#'     def result():
#'         input.compute()    # Take a dependency on the button
#'         
#'         with reactive.isolate():
#'             return f"Result: {input.n()}"
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' Reactive values
#' ===
#' 
#' - reactive.Value: an intermediate variable in Shiny app
#'     - @reactive.Effect: declare that there is a reactive effect
#'     - x = reactive.Value()
#'         - x() or x.get(): get values
#'         - x.set(): to use this function to set values
#' 
#' 
#' ```
#' from shiny import App, reactive, render, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_action_button("toggle", "Toggle value"),
#'     ui.output_text_verbatim("txt"),
#' )
#' 
#' def server(input, output, session):
#'     x = reactive.Value(True)
#' 
#'     @reactive.Effect
#'     @reactive.event(input.toggle)
#'     def _():
#'         x.set(not x())
#' 
#'     @output
#'     @render.text
#'     def txt():
#'         return str(x())
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' Example of reactive values (1)
#' ===
#' 
#' - user_provided_values is not a reactive value
#' 
#' ```
#' from shiny import App, reactive, render, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_numeric("x", "Enter a value to add to the list:", 1),
#'     ui.input_action_button("submit", "Add Value"),
#'     ui.p(
#'         ui.output_text_verbatim("out")
#'     ),
#' )
#' 
#' def server(input, output, session):
#'     # Stores all the values the user has submitted so far
#'     user_provided_values = []
#' 
#'     @reactive.Effect
#'     @reactive.event(input.submit)
#'     def add_value_to_list():
#'         values = user_provided_values()
#'         values.append(input.x())
#' 
#'     @output
#'     @render.text
#'     def out():
#'         return f"Values: {user_provided_values()}"
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' Example of reactive values (2)
#' ===
#' 
#' - user_provided_values can only be modified by set
#' 
#' 
#' ```
#' from shiny import App, reactive, render, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_numeric("x", "Enter a value to add to the list:", 1),
#'     ui.input_action_button("submit", "Add Value"),
#'     ui.p(
#'         ui.output_text_verbatim("out")
#'     ),
#' )
#' 
#' def server(input, output, session):
#'     # Stores all the values the user has submitted so far
#'     user_provided_values = reactive.Value([])
#' 
#'     @reactive.Effect
#'     @reactive.event(input.submit)
#'     def add_value_to_list():
#'         values = user_provided_values()
#'         values.append(input.x())
#' 
#'     @output
#'     @render.text
#'     def out():
#'         return f"Values: {user_provided_values()}"
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' Example of reactive values (3)
#' ===
#' 
#' - a working example
#' 
#' ```
#' from shiny import App, reactive, render, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.input_numeric("x", "Enter a value to add to the list:", 1),
#'     ui.input_action_button("submit", "Add Value"),
#'     ui.p(
#'         ui.output_text_verbatim("out")
#'     ),
#' )
#' 
#' def server(input, output, session):
#'     # Stores all the values the user has submitted so far
#'     user_provided_values = reactive.Value([])
#' 
#'     @reactive.Effect
#'     @reactive.event(input.submit)
#'     def add_value_to_list():
#'         user_provided_values.set(user_provided_values() + [input.x()])
#' 
#'     @output
#'     @render.text
#'     def out():
#'         return f"Values: {user_provided_values()}"
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' page layouts
#' ===
#' 
#' - a common setup for shiny apps:
#'     - page_* is the outermost piece. page_fluid means it will expand to fill the full width of the browser window, rather than stopping at a certain width.
#'     - layout_* positions the pieces inside it (e.g. put them side-by-side).
#'     - panel_* is used for a range of common pieces used in shiny apps.
#'     
#' ```
#' app_ui = ui.page_fluid(
#'     ui.panel_title(),
#'     ui.layout_sidebar(
#'         ui.panel_sidebar(
#'             ...
#'         ),
#'         ui.panel_main(
#'             ...
#'         ),
#'     ),
#' )
#' ```
#' 
#' ![](../figure/layout.png){width=50%}
#' 
#' 
#' Page with sidebar
#' ===
#' 
#' ```
#' from shiny import App, render, ui
#' import matplotlib.pyplot as plt
#' import numpy as np
#' 
#' app_ui = ui.page_fluid(
#'     ui.panel_title("Simulate a normal distribution"),
#' 
#'     ui.layout_sidebar(
#' 
#'       ui.panel_sidebar(
#'         ui.input_slider("n", "Sample size", 0, 1000, 250),
#'         ui.input_numeric("mean", "Mean", 0),
#'         ui.input_numeric("std_dev", "Standard deviation", 1),
#'         ui.input_slider("n_bins", "Number of bins", 0, 100, 20),
#'       ),
#' 
#'       ui.panel_main(
#'         ui.output_plot("plot")
#'       ),
#'     ),
#' )
#' 
#' 
#' def server(input, output, session):
#' 
#'     @output
#'     @render.plot(alt="A histogram")
#'     def plot():
#'         x = np.random.normal(input.mean(), input.std_dev(), input.n())
#' 
#'         fig, ax = plt.subplots()
#'         ax.hist(x, input.n_bins(), density=True)
#'         return fig
#' 
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' 
#' Adding rows and columns
#' ===
#' 
#' ```
#' from shiny import App, ui
#' 
#' style="border: 1px solid #999;"
#' 
#' app_ui = ui.page_fluid(
#'     ui.row(
#'         ui.column(4, "row-1 col-1", style=style),
#'         ui.column(8, "row-1 col-2", style=style),
#'     ),
#'     ui.row(
#'         ui.column(6, "row-2 col-1", style=style),
#'         ui.column(6, "row-2 col-2", style=style),
#'     ),
#' )
#' 
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' Styling Shiny apps
#' ===
#' 
#' - 25 themes in shinyswatch: https://github.com/posit-dev/py-shinyswatch/blob/main/shinyswatch/theme.py
#'     - Dark theme: shinyswatch.darkly()
#'     - Minty theme: shinyswatch.theme.minty()
#'     - Sketchy theme: shinyswatch.theme.sketchy()
#'     - Superhero theme: shinyswatch.theme.superhero()
#' 
#' 
#' ```
#' from shiny import App, Inputs, Outputs, Session, render, ui
#' 
#' import shinyswatch
#' 
#' app_ui = ui.page_fluid(
#'     shinyswatch.theme.darkly(),
#'     ui.input_slider("num", "Number:", min=10, max=100, value=30),
#'     ui.output_text_verbatim("slider_val"),
#' )
#' 
#' 
#' def server(input: Inputs, output: Outputs, session: Session):
#'     @output
#'     @render.text
#'     def slider_val():
#'         return f"{input.num()}"
#' 
#' 
#' app = App(app_ui, server)
#' ```
#' 
#' - customize your own style (if you know about css):
#' 
#' ```
#' ui.page_fluid(
#'     ui.include_css("my-styles.css")
#'   )
#' ```
#' 
#' 
#' Tabs and navigation
#' ===
#' 
#' - a navset_*() style container that determines what the navigation will look like.
#' - nav_*() elements that create different pieces of content.
#'     - Tabs: ui.navset_tab
#'     - Pills: ui.navset_pill
#'     - Lists: ui.navset_pill_list
#'     - Tabs: ui.navset_tab
#' 
#' ```
#' from shiny import App, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.navset_tab(
#'         ui.nav("a", "tab a content"),
#'         ui.nav("b", "tab b content"),
#'         # ui.nav_control("c"), ## no content
#'     )
#' )
#' 
#' app = App(app_ui, None)
#' ```
#' 
#' 
#' Menus
#' ===
#' 
#' ```
#' from shiny import App, ui
#' 
#' app_ui = ui.page_fluid(
#'     ui.navset_tab_card(
#'        ui.nav("a", "tab a content"),
#'        ui.nav_menu(
#'            "Other links",
#' 
#'            # body of menu
#'            ui.nav("b", "tab b content"),
#'            "Plain text",
#' 
#'            # create a horizontal line
#'            "----",
#' 
#'            "More text",
#'            align="right",
#'        ),
#'   )
#' )
#' 
#' app = App(app_ui, None)
#' 
#' ```
#' 
#' References:
#' ===
#' 
#' - https://shiny.posit.co/py/docs/overview.html
#' - https://shiny.posit.co/py/docs/install.html
#' 
