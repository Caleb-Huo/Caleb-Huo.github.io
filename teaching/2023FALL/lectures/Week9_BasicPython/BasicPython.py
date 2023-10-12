#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Tuesday Oct 10th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Basics of Python programming"
#' ---
#' 
## ----setup, include=FALSE-----------------------------------------------------
library(reticulate)
use_python("/usr/local/bin/python3.11")

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
#' - create an environment for the first time
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
#' Review Jupyter Notebook markdown (1)
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
#' Review Jupyter Notebook markdown (2)
#' ===
#' 
#' - import an external image: 
#' 
#' ```
#' ![](https://caleb-huo.github.io/teaching/2023FALL/logo.png){width=50%}
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
#' Review Jupyter Notebook markdown (3)
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
## type(1)

#' - floating point numbers
## type(3.14)

#' - strings
## type("Hello")

#' - boolean type
## type(True)

#' ```
#' type(False)
#' ```
#' 
#' 
#' Conversion among 4 basic data types
#' ===
#' 
## int(1.1)

## str(1.1)

## float("3.15")

## int(True)

## bool(0)

## str(False)

#' 
#' 
#' 3 basci python data structure
#' ===
#' 
#' - list
#' - tuple
#' - dictionary
#' 
#' 
#' list
#' ===
#' 
## names = ["Alice", "Beth", "Carl"]

## len(names)

## names[0]

## names[-1]

#' 
#' - list is mutable
#' 
## names[0] = "Alex"

## names

#' 
#' Tuple
#' ===
#' 
## names = ("Alice", "Beth", "Carl")

## names2 = "Alice", "Beth", "Carl"

## len(names)

## names[0]

## names[-1]

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
## a = [3,1]

## tuple(a)

## 

## b = (3,1)

## list(b)

## 

## sorted(a)

## max(a)

## min(b)

#' 
#' 
#' subsetting of list
#' ===
#' 
#' - the element of a python list can be anything, even a list
#' 
#' 
## data = [[1,2,3], [2,3,4]]

## len(data)

## data[1]

## data[1][1]

#' 
#' 
#' 
#' dictionary
#' ===
#' 
#' - create a dictionary
#' 
## phonebook = {"Alice": 2341,

##             "Beth": 4971,

##             "Carl": 9401

## }

## phonebook

## phonebook["Alice"]

#' 
#' 
## items = [("Name","Smith"), ("Age", 44)]

## d = dict(items)

## d

#' 
## d = dict(Name="Smith", Age=44)

## d

#' 
#' 
#' 
#' Basic python operators
#' ===
#' 
## 2*4

## 2**4 ## raise to the power

## 7//4

## "Hello" + "World"

## False and True

## False or True

#' 
#' ---
#' 
## a = 10

## a = a * 5

## print(a)

#' 
## a = 10

## a *= 5

## print(a)

#' 
#' 
#' Basic string operators
#' ===
#' 
## a = "greetings"

## len(a)

## a[0]

## a[len(a) - 1]

## a[0:4]

## a[4]

#' 
#' - The slicing (e.g., 0:4) is **exclusive**, it doesn't include the last number n. 
#' - For example, a[0:4] will produce a[0], a[1], a[2], and a[3]
#' 
#' Basic string operators
#' ===
#' 
#' 
## a = "Hello"

## a = a + "World"

## print(a)

#' 
#' 
## a = "Hello"

## a += "World"

## print(a)

#' 
#' 
#' Basic string operators
#' ===
#' 
## a = "greetings"

## a[1:]

## a[:3]

## a[:-1]

## a[0:7:2] ## step size 2, default is 1

## a[-1:0:-1]

## a[::-1]

#' 
#' ---
#' 
## b = "Hi"

## b * 3

## name = "Lucas"

## b + " " + "Lucas" + ", " + a

#' 
#' - new line: \\n
#' 
## res = b + " " + "Lucas" + "\n" + a

## print(res)

#' 
#' 
#' python comments
#' ===
#' 
#' - comments: ##
#' 
## 1+10 ## here is a comment

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
#' - print will convert "\\n" to a new line
#' 
#' 
#' 
## a = """

## XXX

## XXXX

## """

## 

## a

## 

## print(a)

#' 
#' python string formatting
#' ===
#' 
## name = 'Alex'

## age = 27

## 

## print('%s is %d years old' % (name, age))

## print('{} is {} years old'.format(name, age))

## print(f'{name} is {age} years old')

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
## bags = 4

## apples_in_bag = 10

## 

## print(f'There are total of {bags * apples_in_bag} apples')

#' 
#' 
## print(f'There are {bags} bags, and {apples_in_bag} apples in each bag.\nSo there are a total of {bags * apples_in_bag} apples')

#' 
#' 
#' f-string formatting float (precision)
#' ===
#' 
#' - precision
#'     - The precision is a value that goes right after the dot character.
#' 
## val = 12.3

## 

## print(f'{val:.2f}')

## print(f'{val:.5f}')

#' 
#' f-string formatting float (width)
#' ===
#' 
#' - width
#'     - The width specifier sets the width of the value. 
#'     - The value may be filled with spaces if the value is shorter than the specified width.
#' 
## val = 12.3

## 

## print(f'{val:2}')

## print(f'{val:7}')

## print(f'{val:07}')

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
## f'{5.5:10.3f}'

#' 
## from math import pi ## will introduce more about python module in the fuction lecture

## pi

## f'{pi:10.6f}'

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
#' Reference
#' ===
#' 
#' - https://docs.python.org/3/index.html
#' - https://github.com/Apress/beginning-python-3ed
#' 
