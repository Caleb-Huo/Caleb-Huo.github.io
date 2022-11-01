
# import tkinter
from tkinter import *

window = Tk()
window.title("Fahrenheit to Celsius Converter")
# window.minsize(width=500, height=800)
window.config(padx=30, pady = 30)

# Label
my_label = Label(text=" is equal to ")
my_label.grid(column=0, row=1)

# Entry
input = Entry(width = 10)
input.insert(END, string="0")
input.grid(column=1, row=0)

F_label = Label(text="°F")
F_label.grid(column=2, row=0)

C_label = Label(text="°C")
C_label.grid(column=2, row=1)


#Text
text = Text(width=13, height=1.1)
text.focus()
text.insert(END, "0")
text.config(state='disabled')
text.grid(column=1, row=1)

def button_clicked():
    new_text = input.get()
    f_tem = float(new_text)
    c_tem = 5 / 9 * (f_tem - 32)
    text.config(state='normal')
    text.delete("1.0", END)
    text.insert(END, str(round(c_tem, 2)))
    text.config(state='disabled')

button = Button(text = "Calculate",command = button_clicked)
button.grid(column=1, row=2)

window.mainloop()

