#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Oct 17th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Basics of Python programming"
#' ---
#' 

#' 
#' 
#' 
#' Outlines
#' ===
#' 
#' - Jupyter Notebook
#' - Three ways to run python code
#' - Basic python data type and structure
#' - Basic python operators
#' - f-strings
#' - Basic string operators
#' - Control flows
#' - Loops
#' - file operations
#' - exceptions
#' - datetime
#' 
#' 
#' Jupyter Notebook
#' ===
#' 
#' ![](../figure/1200px-Jupyter_logo.svg.png){width=30%}
#' 
#' - Open source web application 
#' - Share documents that contain code, results, visualizations etc.
#' - Python version of R markdown
#' - In class demonstration will be performed in Jupyter Notebook
#' - HW will be released in Jupyter Notebook format
#' 
#' 
#' How to start Jupyter Notebook
#' ===
#' 
#' Recommended approach in this class
#' 
#' - Install Anaconda https://www.anaconda.com
#' - Open Anaconda
#' - Select JupyterLab
#' 
#' ![](../figure/anaconda.png){width=60%}
#' 
#' 
#' 
#' 
#' 
#' How to Run Jupyter Notebook
#' ===
#' 
#' ![](../figure/jupyter.png){width=60%}
#' 
#' - run code: 
#'     - shift + enter
#'     - click run
#' - run markdown
#'     - select markdown instead of code
#'     - syntax is very similar to Rmarkdown
#'     - click the chunk, then press m
#' - file system
#'     - on the left hand side
#' 
#' 
#' Jupyter Notebook markdown (1)
#' ===
#' 
#' - header
#' 
#' ```
#' # Header 1
#' ## Header 2
#' ### Header 3
#' #### Header 4
#' ##### Header 5
#' ###### Header 6
#' ```
#' 
#' - Markdown Next
#' 
#' 
#' ```
#' Plain text  
#' End a line with two spaces to start a new paragraph.  
#' *italics* and _italics_  
#' **bold** and __bold__  
#' <span style="color:red">color</span>  
#' superscript^2^  
#' ~~strikethrough~~  
#' [link](www.rstudio.com)   
#' ```
#' 
#' Jupyter Notebook markdown (2)
#' ===
#' 
#' - import an external image: 
#' 
#' ```
#' ![](https://caleb-huo.github.io/teaching/2022FALL/logo.png){width=50%}
#' ```
#' 
#' - inline equation, following latex format [http://www.codecogs.com/latex/eqneditor.php](http://www.codecogs.com/latex/eqneditor.php)
#' 
#' ```
#' $A = \pi \times r^{2}$
#' ```
#' - horizontal rule (or slide break): 
#' 
#' ```
#' ***
#' ```
#' 
#' Jupyter Notebook markdown (3)
#' ===
#' 
#' - Unordered list
#' 
#' ```
#' * unordered list
#' * item 2
#'     + sub-item 1
#'     + sub-item 2
#' ```
#' 
#' - Ordered list
#' ```
#' 1. ordered list
#' 2. item 2
#'     + sub-item 1
#'     + sub-item 2
#' ``` 
#' 
#' 
#' 
#' Three ways to run python code
#' ===
#' 
#' ![](../figure/options_anaconda.png){width=40%}
#' 
#' - python console
#' ```{}
#' print("hello")
#' ```
#' 
#' - Jupyter notebook
#' 
#' ```
#' print("hello")
#' ```
#' 
#' - run a standalone python script (e.g., hello.py) in terminal
#' 
#' ```
#' python hello.py
#' ```
#' 
#' 4 basic data type
#' ===
#' 
#' - integer
type(1)
#' - floating point numbers
type(3.14)
#' - strings
type("Hello")
#' - boolean type
type(True)
#' ```
#' type(False)
#' ```
#' 
#' 
#' Conversion among 4 basic data types
#' ===
#' 
int(1.1)
str(1.1)
float("3.15")
int(True)
bool(0)
str(False)
#' 
#' 
#' list
#' ===
#' 
names = ["Alice", "Beth", "Carl"]
len(names)
names[0]
names[-1]
#' 
#' - list is mutable
#' 
names[0] = "Alex"
names
#' 
#' Tuple
#' ===
#' 
names = ("Alice", "Beth", "Carl")
names2 = "Alice", "Beth", "Carl"
len(names)
names[0]
names[-1]
#' 
#' - Tuple is not mutable
#' 
#' ```
#' names[0] = "Alex"
#' names
#' ```
#' 
#' list and tuple
#' ===
#' 
#' - convert between list and tuple
#' 
a = [3,1]
tuple(a)

b = (3,1)
list(b)

sorted(a)
max(a)
min(b)
#' 
#' subsetting of list
#' ===
#' 
#' - the element of a python list can be anything, even a list
#' 
#' 
data = [[1,2,3], [2,3,4]]
len(data)
data[1]
data[1][1]
#' 
#' 
#' 
#' Basic python operators
#' ===
#' 
2*4
2**4 ## raise to the power
7//4
"Hello" + "World"
False and True
False or True
#' 
#' Basic string operators
#' ===
#' 
a = "greetings"
len(a)
a[0]
a[len(a) - 1]
a[0:4]
a[4]
#' 
#' - The slicing (e.g., 0:4) is **exclusive**, it doesn't include the last number n. 
#' - For example, a[0:4] will produce a[0], a[1], a[2], and a[3]
#' 
#' Basic string operators
#' ===
#' 
a = 10
a = a * 5
print(a)
#' 
a = 10
a *= 5
print(a)
#' 
a = "Hello"
a += "World"
print(a)
#' 
#' 
#' Basic string operators
#' ===
#' 
a = "greetings"
a[1:]
a[:3]
a[:-1]
a[0:7:2] ## step size 2, default is 1 
a[-1:0:-1]
a[::-1]
#' 
#' ---
#' 
b = "Hi"
b * 3
name = "Lucas"
b + " " + "Lucas" + ", " + a
#' 
#' - new line: \\n
#' 
res = b + " " + "Lucas" + "\n" + a
print(res)
#' 
#' 
#' python comments
#' ===
#' 
#' - comments: ##
#' 
1+10 ## here is a comment
#' 
#' - Below is the syntax for a long string
#' 
#' ```
#' """
#' XXX
#' XXXX
#' """
#' ```
#' 
#' - This is also used for comments with more than 1 lines.
#' - print will convert "\n" to a new line
#' 
#' 
#' 
a = """
XXX
XXXX
"""

a

print(a)
#' 
#' python string formatting
#' ===
#' 
name = 'Alex'
age = 27

print('%s is %d years old' % (name, age))
print('{} is {} years old'.format(name, age))
print(f'{name} is {age} years old') 
#' 
#' We will focus on the f-string in this class 
#' 
#' ```
#' f'{}'
#' ```
#' 
#' 
#' Python f-string expressions
#' ===
#' 
#' - We can put expressions between the {} brackets.
#' 
bags = 4
apples_in_bag = 10

print(f'There are total of {bags * apples_in_bag} apples')
#' 
#' 
print(f'There are {bags} bags, and {apples_in_bag} apples in each bag.\nSo there are a total of {bags * apples_in_bag} apples')
#' 
#' 
#' f-string formatting float (precision)
#' ===
#' 
#' - precision
#'     - The precision is a value that goes right after the dot character.
#' 
val = 12.3

print(f'{val:.2f}')
print(f'{val:.5f}')
#' 
#' f-string formatting float (width)
#' ===
#' 
#' - width
#'     - The width specifier sets the width of the value. 
#'     - The value may be filled with spaces if the value is shorter than the specified width.
#' 
val = 12.3

print(f'{val:2}')
print(f'{val:7}')
print(f'{val:07}')
#' 
#' ```{}
#' for x in range(1, 11):
#'     print(f'{x:02} {x*x:3} {x*x*x:4}')
#' ```
#' 
#' f-string formatting float 
#' ===
#' 
#' - precision + width
#' 
#' ```
#' f'{value:{width}.{precision}}'
#' ```
#' 
f'{5.5:10.3f}'
#' 
from math import pi ## will introduce more about python module in the fuction lecture
pi
f'{pi:10.6f}'
#' 
#' - comma separater for larger number
#' ```
#' f'{pi*100000:,.2f}'
#' ```
#' 
#' Input function
#' ===
#' 
#' - Input function
#'     - Allows input from users.
#' 
#' ```{}
#' username = input("What is your name?")
#' print("Hello " + username)
#' ```
#' 
#' 
#' - f string
#' 
#' ```{}
#' print(f"Hello {username}")
#' ```
#' 
#' - Another example:
#' 
#' ```{}
#' num1 = input("First number:")
#' num2 = input("Second number:")
#' res = int(num1) + int(num2)
#' print(res)
#' print(f"{num1} plus {num2} is {res}")
#' ```
#' 
#' 
#' 
#' 
#' Basic string operators
#' ===
#' 
#' - find
#' - index
#' - count
#' - join
#' - split
#' - lower
#' - upper
#' - title
#' - replace
#' - strip
#' 
#' find
#' ===
#' 
title = "I love introduction to Biostatistical computing!"
title.find("I")
title.find("love")
title.find("o")
title.find("o")
title.find("o", 4) ## starting searching index is 4
title.find("XX")
#' 
#' ---
#' 
#' - find() and index() are identical except when not found
#'     - find() produces -1 
#'     - index() produces an error
#' 
#' ```
#' title.index("love")
#' title.index("XX")
#' ```
#' 
#' pattern detection
#' ===
#' 
title = "I love introduction to Biostatistical computing!"
"love" in title
"computing" in title
"XX" in title
title.endswith("computing!")
title.startswith("I love")
title.count("l")
#' 
#' 
#' join
#' ===
#' 
seq = ["1", "2", "3", "4", "5"]
sep = "+"
sep.join(seq)
"".join(seq)
#' 
dirs =( "", "usr", "bin", "env")
"/".join(dirs)
sep = "+"
print("C:" + "\\".join(dirs)) ## single \ has special meaning: treating special symbol as regular symbol 
#' 
#' split
#' ===
#' 
#' - reverse operator of join.
#' 
longSeq = "1+2+3+4+5"
longSeq.split("+")
longSeq.split("3")
#' 
"Using the default value".split()
#' 
#' 
#' lower, upper, title
#' ===
#' 
sentence = "I like introduction to biostatistical computing!"
sentence.lower()
sentence.upper()
sentence.title()
#' 
#' ```
#' sentence.islower()
#' sentence.isupper()
#' sentence.istitle()
#' ```
#' 
#' 
#' strip
#' ===
#' 
#' - removes any leading (whitespace at the beginning) and trailing (whitespace at the end) characters.
#' - whitespace is the default leading character to remove
#' - internal whitespace is kept
#' 
a = "   internal   whitespace is kept     "
a.strip()

b = "*** SPAM * for * everyone!!! ***"
b.strip(" *!")
#' 
#' - also works for new line \n
#' 
c = "\na\nb\n\n\nc\n\n"
c.strip()
#' 
#' replace
#' ===
#' 
a = "This is a cat!"
a.replace("This", "That")
a.replace("is", "eez")
#' 
#' 
#' Control flows
#' ===
#' 
#' ```
#' name = input("What is your name? ")
#' if name.endswith("Smith"):
#'     print("Hello, Mr. Smith")
#' ```
#' 
#' - colon(:) at the end of the if line
#' - indent (e.g., 2 (or 4) whitespaces) before the chunk of code to be executed
#' - In Python if control folow, we don't have parentheses. Indentation is used to determine the end of the code chunk.
#' 
#' ```
#' name = input("What is your name? ")
#' if name.endswith("Smith"):
#'     print("Hello, Mr. Smith")
#'     print("Have a good night")
#' ```
#' 
#' Indentation
#' ===
#' 
#' - Indentation serves another purpose other than code readability
#' - Python treats the statements with the same indentation level (statements with an equal number of whitespaces before them) as a single code block.
#' - Commonly used indent
#'     - 2 whitespaces
#'     - 4 whitespaces
#'     - 1 tab (not recommended)
#' - **This rule of identation is used for flow control, loops, functions etc.**
#' 
#' 
#' if else elif
#' ===
#' 
#' ```
#' num = input("Enter a number: ")
#' if num > 0:
#'     print("The number is positive")
#' else:
#'     print("The number is non-positive")
#' ```
#' 
#' 
#' ```
#' num = input("Enter a number: ")
#' if num > 0:
#'     print("The number is positive")
#' elif num < 0:
#'     print("The number is negative")
#' else:
#'     print("The number is zero")
#' ```
#' 
#' 
#' nested if else conditions
#' ===
#' 
#' - the scope of the if else condition is determined by the indent
#' 
#' ```
#' name = input("What is your name? ")
#' if name.endswith("Smith"):
#'     if name.startswith("Mr."):
#'         print("Hello, Mr. Smith")
#'     elif name.startswith("Mrs."):
#'         print("Hello, Mrs. Smith")
#'     else:
#'         print("Hello, Smith")
#' else:
#'     print("Hello, Stranger")
#' ```
#' 
#' if else same line
#' ===
#' 
#' - original
#' 
#' ```
#' number = input("Please enter a number: ")
#' if int(number) % 2 == 0:
#'     print("even")
#' else:
#'     print("odd")
#' ```
#' 
#' - one line version
#' ```
#' number = input("Please enter a number: ")
#' print("even") if int(number) % 2 == 0 else print("odd")
#' ```
#' 
#' 
#' 
#' True or False conditions
#' ===
#' 
1>2
4 == 5
"ap" in "apple"
"apple" in ["apple", "orange"]
True and False
True or False
not True
#' 
#' True or False conditions
#' ===
#' 
#' - we could use < (or >) to connect a series of comparisons
#' 
a = 4
b = 6
c = 9

a < b and b < c

a < b < c

a < c > b
#' 
#' 
#' 
#' match (available for python >= 3.10)
#' ===
#' 
#' - To select one cases from multiple choices
#' 
#' ```reticulate{python}
#' status = 400    
#' 
#' match status:
#'     case 400:
#'         print("Bad request")
#'     case 404:
#'         print("Not found")
#'     case 418:
#'         print("I'm a teapot")
#'     case _:
#'         print("Something's wrong with the internet")
#' ```
#' 
#' 
#' 
#' for loops
#' ===
#' 
words = ["cat", "dog", "gator"]
for w in words:
     print(w)
#' 
#' 
words = ["cat", "dog", "gator"]
for w in words:
     print(f"{w} has {len(w)} letters in it.")
#' 
#' 
#' range() function
#' ===
#' 
for i in range(3):
    print(i)
#' 
#' - The range(n) is **exclusive**, it doesn't include the last number n. 
#' - It creates the sequence of numbers from start to stop -1. 
#' - For example, range(5) will produce [0, 1, 2, 3, 4]
#' 
#' 
list(range(3))
list(range(3,7))
#' 
#' 
#' range() function
#' ===
#' 
#' - range with step size rather than 1.
#' 
list(range(3,8,2))
list(range(7,2,-2))
#' 
#' - range over a list
#' 
words = ["cat", "dog", "gator"]
for i in range(len(words)):
     print(i, words[i])
#' 
#' 
#' break
#' ===
#' 
for num in range(1, 10):
  if num % 5 == 0:
    print(f"{num} can be divided by 5")
    break
  print(f"{num} cannot be divided by 5")
#' 
#' 
#' continue
#' ===
#' 
for num in range(1, 10):
  if num % 5 == 0:
    continue
  print(f"{num} cannot be divided by 5")
#' 
#' 
#' pass
#' ===
#' 
#' - In python, pass is the null statement.
#' - It is just a placeholder for the functionality to be added later.
#' - Pass does nothing.
#' 
sequence = {'p', 'a', 's', 's'}
for val in sequence:
    pass
#' 
a = 33
b = 200

if b > a:
  pass
#' 
#' 
#' 
#' while loop
#' ===
#' 
num = 1
while num<10:
  if num % 5 == 0:
    print(f"{num} can be divided by 5")
    break
  print(f"{num} cannot be divided by 5")
  num+=1
#' 
num = 0
while num<10:
  num+=1
  if num % 5 == 0:
    continue
  print(f"{num} cannot be divided by 5")
#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' file operation (read)
#' ===
#' 
#' - open file, display, and close (release memory)
#' 
file = open("my_file.txt")
contents = file.read()
print(contents)
file.close()
#' 
#' - Alternative approach without closing step
#' 
with open("my_file.txt") as file:
    contents = file.read()
    print(contents)
#' 
#' file operation (read)
#' ===
#' 
#' - read multilple lines, 
#' - save the result in a list
#'     - each element of the list contains a line
#' 
myfile = "my_file.txt"
with open(myfile) as file:
    lines = file.readlines()

for aline in lines:
    print(aline.strip())
#' 
#' 
#' file operation (write)
#' ===
#' 
#' 
#' - write to file (overwrite original file) 
with open("new_file.txt", mode="w") as file:
    file.write("I like biostatistical computing!")
#' 
#' - append to file (append at the end of the original file) 
with open("new_file.txt", mode="a") as file:
    file.write("We like biostatistical computing!")
#' 
#' Exceptions
#' ===
#' 
#' 
#' - FileNotFound
#' ```
#' with open("a_file.txt") as file:
#'     file.read()
#' ```
#' 
#' - IndexError
#' ```
#' fruit_list = ["Apple", "Banana", "Pear"]
#' fruit_list[3]
#' ```
#' 
#' - TypeError
#' ```
#' text = "abc"
#' print(text + 5)
#' ```
#' 
#' - raise an error
#' ```
#' raise TypeError("This is an error that I made up!")
#' ```
#' 
#' 
#' Handle exceptions
#' ===
#' 
#' - The errors (exceptions) are handled by except.
#' - The program will keep executing.
#' 
try:
    file = open("a_file.txt")
    print(1 + "2")
except FileNotFoundError:
    print("Catch FileNotFoundError")
except TypeError as error_message:
    print(f"Here is the error: {error_message}.")
else:
    content = file.read()
    print(content)
finally: ## will happen no matter what happens
    file.close()
    print("File was closed.")
#' 
#' datetime
#' ===
#' 
#' - The datetime module supplies classes for manipulating dates and times.
#' 
import datetime as dt
now = dt.date.today() ## date only
now.year
now.month
now.day
# now.weekday()
#' 
birthday = dt.date(1995, 7, 31)
age = now - birthday
age.days
#' 
#' datetime
#' ===
#' 
now = dt.datetime.now()
now.year
now.month
now.day
now.hour
#' 
#' --- 
#' 
now.minute
now.second
now.microsecond
now.weekday()

#' 
#' datetime
#' ===
#' 
now = dt.datetime.now()

print(f'{now:%Y-%m-%d %H:%M}')
#' 
#' 
#' Reference
#' ===
#' 
#' - https://docs.python.org/3/index.html
#' 
#' 
