#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Thursday Oct 19th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "functions and modules"
#' ---
#' 
## ----setup, include=FALSE----------------------------------------------------------------------------------
library(reticulate)
use_python("/usr/local/bin/python3.10")

#' 
#' 
#' 
#' Outlines
#' ===
#' 
#' - Writing Basic Python functions
#' - Some built-in Python functions
#' - Load Python modules
#' - Function arguments
#' - Lambda function
#' - Scope of a function
#' - Multiple return value
#' - Prepare your own function module
#' - Install Python modules
#' - Python environment (optional)
#' 
#' 
#' Writing Basic Python functions
#' ===
#' 
#' - function without argument
#' 
#' 
## def hello():

##     print('Hello World!')

## 

## hello()

#' 
#' - function with an argument
#' 
## def hello(name):

##     print('Hello, ' + name + "!")

## 

## hello("Lucas")

#' 
#' Function Descriptions and Helps
#' ===
#' 
## def inc(x):

##   """Increase by one

## 

##   Keyword arguments:

##   x -- the base number to be increased

##   """

##   return x + 1

## 

## inc(3)

## inc.__doc__ ## check docstring

## 

#' 
#' - get help page
#' 
#' ```
#' ?inc
#' help(inc)
#' ```
#' 
#' 
#' Function structure
#' ===
#' 
#' - Keyword ""def"" that marks the start of the function 
#' - A colon (:) to mark the end of the function header.
#' - The function body must have same indentation for the same level of code.
#' - (Optional)
#'   - Documentation string (docstring) to describe what the function does.
#'   - Parameters (arguments) to pass values to a function.
#'   - Return statement to return a value from the function.
#' 
#' 
#' 
#' Recursive function
#' ===
#' 
#' - $f(n) = n! = n(n-1)\ldots 1$
#' 
## def factorial(n):

##     result = n

##     for i in range(1,n):

##         result *= i

##     return result

## 

## factorial(5)

#' 
#' - recursive function
#' 
## def factorial(n):

##     if n==1: return 1

##     return n*factorial(n-1)

## 

## factorial(5)

#' 
#' 
#' In class exercise
#' ===
#' 
#' - Write a recursive function to calculate the Fibonacci number
#'   - f(0) = 1
#'   - f(1) = 1
#'   - f(2) = f(1) + f(0) = 2
#'   - f(3) = f(2) + f(1) = 3
#'   - f(4) = f(3) + f(2) = 5
#'   - ...
#'   - f(n) = f(n-1) + f(n-2) 
#' 
#' ---
#' 
## def fibonacci(n):

##     if(n<=1):

##         return(1)

##     else:

##         return(fibonacci(n-1) + fibonacci(n-2))

## 

## fibonacci(4)

## fibonacci(5)

#' 
#' 
#' 
#' Some built-in Python functions
#' ===
#' 
## abs(-4)

## pow(2,3) ## 2**3

## round(3.14)

## max(6,9)

## min([1,2,3,7])

#' 
#' ---
#' 
## sum(range(10))

## help(sum)

#' 
#' ```
#' ?sum ## this way only works for Jupyter Notebook
#' ```
#' 
#' Module
#' ===
#' 
#' - A Python module is similar to an R package
#'     - contains a set of functions.
#' - Create a module    
#'     - put the following function in a file mymod.py
#'     
#' ```
#' def hello(name):
#'     print('Hello ' + name + '!')
#' ```
#' 
#' ```
#' def myadd(a, b):
#'     return(a + b)
#' ```
#' 
#' - In python
## import mymod

## 

## mymod.hello("Lucas")

## mymod.myadd(1, 10)

#' 
#' 
#' Module
#' ===
#' 
#' - in addition to functions, a module can also contain data variables.
#'     - put the following code in the file mymod.py
#' 
#' ```
#' Student1 = {
#'   "name": "Amy",
#'   "age": "23",
#'   "country": "Canada"
#' }
#' ```
#' 
#' 
#' - In python
## import mymod

## 

## astudent = mymod.Student1

## print(astudent)

## for akey in astudent:

## 		print(akey + ": " + astudent[akey])

#' 
#' 
#' 
#' 
#' math modules
#' ===
#' 
#' After import module, all the objects in the module are available via module.object
#' 
## import math

## 

## math.sqrt(3)

## math.floor(10.4)

## math.ceil(10.4)

## math.log(16,2)

## math.log2(16)

## math.pi

#' 
#' math modules
#' ===
#' 
#' - create a shortcut name for the module
#' 
## import math as m

## m.pi

#' 
#' - import a specific object from a module
#' 
## from math import pi

## pi

#' 
#' - import multiple objects from a module
#' 
## from math import pi, log2

## log2(pi)

#' 
#' 
#' os module
#' ===
#' 
#' - os module contains functions to interacting with the operation system
#' 
## import os

#' 
#' - get current working directory
#' 
## cwd = os.getcwd() ## get current working directory

## print("Current working directory is " + cwd)

#' 
#' - list all files
#' 
## os.listdir()

## ## os.listdir(cwd)  same as this one

#' 
#' ---
#' 
#' - set working directory
#' 
## os.chdir('..') ## go back one folder

## os.getcwd()

## os.chdir(cwd) ## go back to cwd

## os.getcwd()

#' 
#' - create folders
#' 
#' ```
#' os.mkdir("tmp") ## create a single folder
#' os.makedirs("foo/bar")  ## create a recursive folder
#' ```
#' 
#' 
#' - remove files/folders
#' 
#' ```
#' os.remove(afile) ## remove afile
#' os.rmdir(afolder) ## remove a folder
#' ```
#' 
#' ---
#' 
#' 
#' - join path
#' 
## file = 'Untitled.ipynb'

## 

## # File location

## location = "/Users/zhuo/Desktop/" ## change this to your own directory

## os.listdir(location)

## 

## # Path

## path = os.path.join(location, file)

## print(path)

#' 
#' ---
#' 
#' - file/folder exists
#' 
## os.path.exists(location)

## os.path.exists(path)

## os.path.isfile(path)

## os.path.isdir(location)

#' 
#' 
#' random module
#' ===
#' 
#' - the random module allows to generate random numbers
#'     - random(): random number between 0 and 1
#' 
## import random

## 

## print(random.random())

#' 
#' - same random seed will allow reproducible results
#' 
## random.seed(10)

## print(random.random())

## 

## random.seed(10)

## print(random.random())

#' 
#' 
#' random integer
#' ===
#' 
#' - randomint(a,b): random integer between a (inclusive) and b(inclusive)
#' 
## print(random.randint(1, 3))

#' 
#' - randomrange(a,b): random integer between a (inclusive) and b(exclusive)
#' 
## print(random.randrange(1, 3))

#' 
#' random choices and shuffle
#' ===
#' 
#' - random.choice: choose one from a list
#' 
## mylist = ["apple", "banana", "cherry"]

## random.choice(mylist)

#' 
#' - random.choices: choose k from a list
#' 
## random.choices(mylist, k = 2)

#' 
#' 
#' - random.shuffle: shuffle the order (inplace operator)
#' 
## random.shuffle(mylist)

## mylist

#' 
#' 
#' Default arguments 
#' ===
#' 
## def add(arg1 = 1, arg2 = 2):

##     return(arg1 + arg2)

## 

## add(arg1 = 3, arg2 = 4)

## add(arg1 = 3)

## add()

#' 
#' 
#' 
#' Function arguments matching rule
#' ===
#' 
## def parrot(arg, brg=["a", "b"], crg='python'):

##     print("arg: " + str(arg))

##     print("brg: " + str(brg))

##     print("crg: " + str(crg))

#' 
## parrot(1000)                             # 1 positional argument

## parrot(arg=1000)                         # 1 keyword argument

## parrot(arg=1000000, crg='R')             # 2 keyword arguments

#' 
#' ---
#' 
#' - additional matching rules
#' 
## parrot(crg='JAVA', arg=1000000)             # 2 keyword arguments

## parrot(32611, [1,2,3], "C++")         # 3 positional arguments

## parrot(567, brg=list("abc"))  # 1 positional, 1 keyword

## # parrot(brg=list("abc"), 567)  # 1 positional, 1 keyword, doesn't work

#' 
#' Extra arguments 
#' ===
#' 
#' - a motivting example
#' 
#' - two input
#' 
#' ```
#' def add(a,b):
#'   return(a + b)
#' ```
#' 
#' - four input
#' 
#' ```
#' def add(a,b,c,d):
#'   return(a + b + c + d)
#' ```
#' 
#' - how about the number of input is unknown?
#' 
#' 
#' 
#' Extra arguments 
#' ===
#' 
#' - *params: any extra arguments for the function
#'     - pass as tuple
#' 
## def print_args(*args):

##     print(args)

## 

## print_args('Testing') ## tuple of length 1

## print_args(1,2,3)

#' 
## def print_args_2(title, *args):

##     print(title)

##     print(args)

## 

## print_args_2("Args:", 1,2,3)

## print_args_2("Nothing:")

#' 
#' In class exercise
#' ===
#' 
#' - Write a function add(), which can take an arbitrary number of argument input, 
#' - The function will return the sum of these argument
#' - The result we would expect:
#' 
#' ```
#' add(1,2,3,4,5) # 15
#' add(1,2,3,4,5,6,7,8,9,10) # 55
#' ```
#' 
#' ---
#' 
## def add(*args):

##     print(type(args)) ## the input args forms a tuple

##     sum = 0

##     for n in args:

##         sum += n

##     return sum

## 

## add(1,2,3,4,5) # 15

## add(1,2,3,4,5,6,7,8,9,10) # 55

#' 
#' 
#' Extra keywords
#' ===
#' 
#' - must specify the name of the keywords (e.g., x=1)
#'     - pass as dict
#'     
## def print_keywords_3(**keywords):

##     print(keywords)

## 

## print_keywords_3(x=1,y=2,z=3)

#' 
## def print_keywords_4(**keywords):

##     for kw in keywords:

##         print(kw, ":", keywords[kw])

## print_keywords_4(x=1,y=2,z=3)

#' 
#' 
#' ---
#' 
#' 
#' 
## def print_params_4(x,y,z=3,*pospar, **keypar):

##     print(x,y,z)

##     print(pospar)

##     print(keypar)

## 

## print_params_4(x=1,y=2)

## print_params_4(1,2,3,5,6,7,foo=1)

#' 
#' 
#' Assertion
#' ===
#' 
#' - assertion:
#'     - the *assert* keyword
#'     - the expression/condition to test
#'     - an optional message
#'     - put assertion at the beginning of a function to verify arguments
#' 
#' ```
#' assert expression[, assertion_message]
#' ```
#' 
## number = 1 ## try -1 as well

## assert number > 0

## assert number > 0, f"number greater than 0 expected, got: {number}"

#' 
#' - can also integrate with try-expect statement to capture "AssertionError"
#' 
#' Lambda function
#' ===
#' 
#' - a small anonymous function.
#' - can take any number of arguments, but with only one expression.
#' 
## def f1(x):

##     return(x + 10)

## 

## f1(5)

## 

## f2 = lambda x : x + 10

## 

## f2(5)

## 

## (lambda x : x + 10)(5)

#' 
#' 
#' Lambda function with more than 1 argument
#' ===
#' 
## def f3(a,b):

##   return(a*b)

## 

## f3(5,6)

## 

## f4 = lambda a, b : a * b

## print(f4(5, 6))

#' 
#' 
#' Function scopes
#' ===
#' 
#' - Global Scope
#'     - A variable created in the main Python code belongs to the global scope.
#'     - Global variables are available for both global and local.
#' 
## x = 300

## 

## def afun():

##   print(x)

## 

## afun()

## print(x)

#' 
#' 
#' Function scopes
#' ===
#' 
#' - Same variable names
#'     - If you have the same variable name inside and outside of a function, 
#'         - one available in the global scope (outside the function)
#'         - one available in the local scope (inside the function):
#' 
## x = 300

## 

## def afun():

##   x = 200

##   print(x)

## 

## afun()

## print(x)

#' 
#' Function scopes
#' ===
#' 
#' - Global Keyword
#'     - If you want a local variable to be in the global scope, you may want to use the global keyword
#' 
#' 
## def bfun():

##   global x

##   x = 50

## 

## bfun()

## print(x)

#' 
## x = 50

## 

## def bfun():

##   global x

##   x = 60

## 

## bfun()

## print(x)

#' 
#' Multiple return values
#' ===
#' 
## def f23(x):

##     x_square = x**2

##     x_cubic = x**3

##     res_tuple = (x_square, x_cubic)

##     return(res_tuple)

## 

## f23(3)

## 

## a, b = f23(5) ## to receive multiple return values

## print(a)

## print(b)

#' 
#' Other looping technique
#' ===
#' 
#' - parallel iteration
#' 
## d = ["a", "b", "c"]

## e = [1, 2, 3]

## 

## zip(d,e)

## list(zip(d,e))

## 

## for i, j in zip(d,e):

##     print(i + str(j))

#' 
#' - iterable object
#' 
#' ```
#' a = zip(d,e)
#' next(a)
#' ```
#' 
#' ---
#' 
#' - enumerated iteration: a tuple of index and the original object
#' 
## d = ["a", "b", "c"]

## for index, string in enumerate(d):

##     print(string + str(index))

#' 
## list(enumerate(d))

#' 
## x = enumerate(d) ## iterable object

## next(x)

#' 
#' list comprehension
#' ===
#' 
#' - List comprehension provides a short syntax to create a new list based on an existing list
#' 
#' 
## squares = []

## for x in range(10):

##     squares.append(x**2)

## 

## squares

#' 
#' - put everything in one line
#' 
#' 
## [x**2 for x in range(10)]

#' 
#' 
#' list comprehension
#' ===
#' 
#' - two variable list comprehension
#' 
## [(i,j) for i in range(3) for j in range(3)]

#' 
#' 
#' - print even number only
#' 
## [i for i in range(10) if i%2==0 ]

#' 
#' 
#' - subsetting
#' 
## names = ["Amy", "Beth", "Carl"]

## [aname[0] for aname in names]

#' 
#' list comprehension for dictionary
#' ===
#' 
#' - list comprehension for dictionary
#' 
## names = ["Amy", "Beth", "Carl", "Dan", "Emily", "Frank"]

## 

## import random

## 

## students_scores = {name: random.randint(0, 100) for name in names}

## 

## students_scores

#' 
#' 
#' Install new packages
#' ===
#' 
#' - pip
#'     - it is included by default with the Python binary installers.
#' - conda
#'     - alternative way to install packages, Conda lets you easily switch between environment on your local computer
#' 
#' pip install packages
#' ===
#' 
#' - Install in jupyter notebook (best practice to avoid confusion)
#' 
#' ```
#' python -m pip install numpy
#' import numpy as np ## test if the package has been installed in python
#' python3 -m pip install numpy==1.23 ## specifying a package version
#' python3.9 -m pip install --upgrade numpy ## upgrade a package
#' pip install numpy --user ## if you are not a admin
#' ```
#' 
#' 
#' - macOS or linux user
#'     - in the terminal
#'     - pip == python -m pip
#'     - python (default python), python3 (default version for python3), python 3.9 (python version 3.9)
#' 
#' ```
#' python -m pip install numpy
#' >>> import numpy as np ## test if the package has been installed in python
#' python3 -m pip install numpy==1.23 ## specifying a package version
#' python3.9 -m pip install --upgrade numpy ## upgrade a package
#' pip install numpy --user ## if you are not a admin
#' ```
#' 
#' - windows user
#'     - in the terminal
#' 
#' 
#' ```
#' py -3 -m pip install numpy
#' >>> import numpy as np ## test if the package has been installed in python
#' py -3 -m pip install numpy==1.23 ## specifying a package version
#' py -3 -m pip install --upgrade numpy ## upgrade a package
#' py -3 -m pip install numpy --user ## if you are not a admin
#' ```
#' 
#' - you could change numpy to other Python package name
#' 
#' 
#' 
#' conda install packages
#' ===
#' 
#' - Anaconda
#'     - with over 1500 scientific packages automatically installed at once
#' 
#' - Miniconda (recommended in this class)
#'     - need to install individual packages
#' 
#' - install conda:
#'     - https://docs.conda.io/en/latest/miniconda.html
#' 
#' - after installation    
#' 
#' ```
#' conda install numpy
#' conda install numpy=1.16
#' conda list --name numpy conda ## check current installed version
#' conda update numpy
#' ```
#' 
#' How to prepare your own package, and make it installable using pip
#' ===
#' 
#' - https://packaging.python.org/en/latest/tutorials/packaging-projects/
#' 
#' Python environment 
#' ===
#' 
#' - With environment, 
#'     - you can install different version of Python packages independent of the admin user.
#' 
#' - how to create an environment
#'     - using anaconda software (user friendly interface)
#'     - in the terminal using conda
#' 
#' 
#' ```
#' ## source ~/opt/miniconda3/bin/activate ## need this for macOS Catalina or later
#' conda create -n myenv
#' ## conda create -n myenv python=3.9 ## create an environment with a specific Python version.
#' conda activate myenv ## activate the environment
#' conda info --envs ## list all environment
#' conda install -n myenv scipy=0.17.3
#' python ## open python with this environment
#' conda deactivate
#' ```
#' 
#' 
#' Python environment (using pip)
#' ===
#' 
#' - https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/26/python-virtual-env/
#' 
#' 
#' Reference
#' ===
#' 
#' - https://docs.python.org/3/index.html
#' - https://github.com/Apress/beginning-python-3ed
#' - https://www.programiz.com/python-programming/function
#' - https://www.geeksforgeeks.org/os-module-python-examples/
#' - https://www.w3schools.com/python/python_modules.asp
#' - https://docs.python.org/3/installing/index.html
#' - https://towardsdatascience.com/manage-your-python-virtual-environment-with-conda-a0d2934d5195
#' 
