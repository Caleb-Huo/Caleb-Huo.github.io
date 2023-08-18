#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday Nov 2nd, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Python graphical user interface via Tinker"
#' ---
#' 

#' 
#' Outline
#' ===
#' 
#' - Introduction to GUI and Tkinter
#' - Basics about Tkinter
#' - grid for Tkinter
#' - a temperature conversion application
#' 
#' 
#' 
#' GUI
#' ===
#' 
#' - GUI stands for graphical user interface.
#' - Tkinter is a popular python GUI, which is the standard Python interface to the Tcl/Tk GUI toolkit.
#'   - Tcl: (Tool Command Language) is a very powerful but easy to learn dynamic programming language, suitable for a very wide range of uses, including web and desktop applications, networking, administration, testing and many more.
#'   - Tk: is a graphical user interface toolkit that takes developing desktop applications to a higher level than conventional approaches. Tk is the standard GUI not only for Tcl, but also for other applications.
#' - Advantage of Tkinter:
#'   - Tkinter is most likely already installed in a recent version of Python
#'   - A vast collection of well known widgets are available in Tkinter, including all the most common ones like buttons, labels, checkboxes etc.
#'   - Code to each widget is straightforward
#' 
#' 
#' Basics about Tkinter
#' ===
#' 
#' - package: tkinter
#' - initialize a window object from the Tk class.
#' - use the mainloop() to maintain the window open
#' - the rest of the widget code should be inside of the following chunk of code
#' 
#' ```
#' from tkinter import *
#' 
#' window = Tk()
#' window.title("My first GUI progrom")
#' window.minsize(width=500, height=800)
#' 
#' window.mainloop()
#' ```
#' 
#' 
#' label
#' ===
#' 
#' - pack is one of the layout, which packs widgets relative to the earlier widget.
#' 
#' ```
#' # Label
#' my_label = Label(text="I am a Label", font = ("Arial", 24, "bold"))
#' my_label.pack()
#' # my_label.pack(side = "left")
#' # my_label.pack(side = "bottom")
#' # my_label.pack(expand = True)
#' 
#' my_label["text"] = "New Text"
#' my_label.config(text="New Text")
#' 
#' ```
#' 
#' 
#' entry
#' ===
#' 
#' ```
#' input = Entry(width = 10)
#' input.insert(END, string="Some text to begin with.")
#' input.pack()
#' print(input.get())
#' ```
#' 
#' button
#' ===
#' 
#' ```
#' def button_clicked():
#'     new_text = input.get()
#'     my_label.config(text=new_text)
#' 
#' button = Button(text = "Click Me",command = button_clicked)
#' button.pack()
#' ```
#' 
#' 
#' text
#' ===
#' 
#' ```
#' #Text
#' text = Text(height=5, width=30)
#' #Puts cursor in textbox.
#' text.focus()
#' #Adds some text to begin with.
#' 
#' text.insert(END, "Example of multi-line text entry.")
#' #Get's current value in textbox at line 1, character 0
#' print(text.get("1.0", END))
#' text.pack()
#' ```
#' 
#' Spinbox (optional)
#' ===
#' 
#' ```
#' def spinbox_used():
#'     #gets the current value in spinbox.
#'     print(spinbox.get())
#' spinbox = Spinbox(from_=0, to=10, width=5, command=spinbox_used)
#' spinbox.pack()
#' ```
#' 
#' 
#' 
#' Scale (optional)
#' ===
#' 
#' ```
#' def scale_used(value):
#'     print(value)
#' scale = Scale(from_=0, to=100, command=scale_used)
#' scale.pack()
#' ```
#' 
#' 
#' 
#' Checkbutton (optional)
#' ===
#' 
#' ```
#' def checkbutton_used():
#'     #Prints 1 if On button checked, otherwise 0.
#'     print(checked_state.get())
#' #variable to hold on to checked state, 0 is off, 1 is on.
#' checked_state = IntVar()
#' checkbutton = Checkbutton(text="Is On?", variable=checked_state, command=checkbutton_used)
#' checked_state.get()
#' checkbutton.pack()
#' ```
#' 
#' 
#' 
#' Radiobutton (optional)
#' ===
#' 
#' ```
#' def radio_used():
#'     print(radio_state.get())
#' #Variable to hold on to which radio button value is checked.
#' radio_state = IntVar()
#' radiobutton1 = Radiobutton(text="Option1", value=1, variable=radio_state, command=radio_used)
#' radiobutton2 = Radiobutton(text="Option2", value=2, variable=radio_state, command=radio_used)
#' radiobutton1.pack()
#' radiobutton2.pack()
#' ```
#' 
#' 
#' Listbox (optional)
#' ===
#' 
#' ```
#' def listbox_used(event):
#'     # Gets current selection from listbox
#'     print(listbox.get(listbox.curselection()))
#' 
#' listbox = Listbox(height=4)
#' fruits = ["Apple", "Pear", "Orange", "Banana"]
#' for item in fruits:
#'     listbox.insert(fruits.index(item), item)
#' listbox.bind("<<ListboxSelect>>", listbox_used)
#' listbox.pack()
#' ```
#' 
#' 
#' layout
#' ===
#' 
#' - pack():
#'   - relative to the earlier widget
#' - place():
#'   - specify the exact coordinate
#' - grid(): (recommended)
#'   - flexible, relative coordinate
#' 
#' 
#' place
#' ===
#' 
#' ```
#' from tkinter import *
#' 
#' window = Tk()
#' window.title("My first GUI progrom")
#' window.minsize(width=500, height=800)
#' window.config(padx=100, pady = 100)
#' 
#' # Label
#' my_label = Label(text="I am a Label", font = ("Arial", 24, "bold"))
#' my_label.pack()
#' my_label.config(padx=100, pady = 100)
#' 
#' my_label["text"] = "New Text"
#' my_label.place(x=0,y=0)
#' ```
#' 
#' grid
#' ===
#' 
#' - grid() locates widgets in a two dimensional grid using row and column absolute coordinates.
#' 
#' ![](../figure/Tkinter-grid-Grid-Geometry.png)
#' 
#' grid
#' ===
#' 
#' ```
#' from tkinter import *
#' 
#' window = Tk()
#' window.title("My first GUI progrom")
#' window.minsize(width=500, height=800)
#' window.config(padx=100, pady = 100)
#' 
#' # Label
#' my_label = Label(text="I am a Label", font = ("Arial", 24, "bold"))
#' # my_label.pack()
#' my_label.config(padx=100, pady = 100)
#' 
#' my_label["text"] = "New Text"
#' my_label.grid(column=0, row=0)
#' ```
#' 
#' 
#' grid
#' ===
#' 
#' - button
#' 
#' ```
#' def button_clicked():
#'     new_text = input.get()
#'     my_label.config(text=new_text)
#' 
#' button = Button(text = "Click Me",command = button_clicked)
#' # button.pack()
#' button.grid(column=1, row=1)
#' ```
#' 
#' - entry
#' 
#' ```
#' input = Entry(width = 10)
#' input.insert(END, string="Some text to begin with.")
#' # input.pack()
#' input.grid(column=2, row=2)
#' 
#' print(input.get())
#' ```
#' 
#' Temperature converter
#' ===
#' 
#' - formula:
#' 
#' $$(F − 32) × 5/9 = C$$
#' 
#' $$(32°F − 32) × 5/9 = 0°C$$
#' 
#' - example output:
#' 
#' ![](../figure/temperature_conversion.png)
#' 
#' - code:
#'   - [https://caleb-huo.github.io/teaching/data/code/temperature_converter.py](temperature_converter.py)
#' 
#' 
#' Reference
#' ===
#' 
#' - https://docs.python.org/3/library/tkinter.html
#' - https://www.udemy.com/course/100-days-of-code/
#' 
