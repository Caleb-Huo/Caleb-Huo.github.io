#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Thursday Oct 12th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Basics of Python programming (part 2)"
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
#' - Control flows
#' - Loops
#' - Basic string operators
#' - file operations
#' - exceptions
#' - datetime
#' - more on python data structure
#'   - list
#'   - dictionary
#'   - tuple
#' 
#' 
#' 
#' Control flows
#' ===
#' 
#' ```
#' num = input("Enter a number: ")
#' if int(num) > 0:
#'     print(f"{num} is positive")
#' ```
#' 
#' - colon(:) at the end of the if line
#' - indent (e.g., 2 (or 4) whitespaces) before the chunk of code to be executed
#' - In Python if control folow, we don't have parentheses like in R. Indentation is used to determine the end of the code chunk.
#' 
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
#' if int(num) > 0:
#'     print(f"{num} is positive")
#' else:
#'     print(f"{num} is not positive")
#' ```
#' 
#' 
#' ```
#' num = input("Enter a number: ")
#' if int(num) > 0:
#'     print("The number is positive")
#' elif int(num) < 0:
#'     print("The number is negative")
#' else:
#'     print("The number is zero")
#' ```
#' 
#' 
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
## 1>2

## 4 == 5

## "ap" in "apple"

## "apple" in ["apple", "orange"]

## True and False

## True or False

## not True

#' 
#' True or False conditions
#' ===
#' 
#' - we could use < (or >) to connect a series of comparisons
#' 
## a = 4

## b = 6

## c = 9

## 

## a < b and b < c

## 

## a < b < c

## 

## a < c > b

#' 
#' 
#' 
#' match (available for python >= 3.10)
#' ===
#' 
#' - To select one cases from multiple choices
#' 
## status = 400

## 

## match status:

##     case 400:

##         print("Bad request")

##     case 404:

##         print("Not found")

##     case 418:

##         print("I'm a teapot")

##     case _:

##         print("Something's wrong with the internet")

#' 
#' 
#' 
#' for loops
#' ===
#' 
## words = ["cat", "dog", "gator"]

## for w in words:

##      print(w)

#' 
#' 
## words = ["cat", "dog", "gator"]

## for w in words:

##      print(f"{w} has {len(w)} letters in it.")

#' 
#' 
#' range() function
#' ===
#' 
## for i in range(3):

##     print(i)

#' 
#' - range(n) creates an iterable object
#' - list(iterable) convert an iterable to a list
#' - The range(n) is **exclusive**, it doesn't include the last number n. 
#' - It creates the sequence of numbers from start to stop -1. 
#' - For example, list(range(5)) will produce [0, 1, 2, 3, 4]
#' 
#' 
## list(range(3))

## list(range(3,7))

#' 
#' 
#' range() function
#' ===
#' 
#' - range with step size rather than 1.
#' 
## list(range(3,8,2))

## list(range(7,2,-2))

#' 
#' - range over a list
#' 
## words = ["cat", "dog", "gator"]

## for i in range(len(words)):

##      print(i, words[i])

#' 
#' 
#' break
#' ===
#' 
## for num in range(1, 10):

##   if num % 5 == 0:

##     print(f"{num} can be divided by 5")

##     break

##   print(f"{num} cannot be divided by 5")

#' 
#' 
#' continue
#' ===
#' 
## for num in range(1, 10):

##   if num % 5 == 0:

##     continue

##   print(f"{num} cannot be divided by 5")

#' 
#' 
#' pass
#' ===
#' 
#' - In python, pass is the null statement.
#' - It is just a placeholder for the functionality to be added later.
#' - Pass does nothing.
#' 
## sequence = {'p', 'a', 's', 's'}

## for val in sequence:

##     pass

#' 
## a = 33

## b = 200

## 

## if b > a:

##   pass

#' 
#' 
#' 
#' while loop
#' ===
#' 
## num = 1

## while num<10:

##   if num % 5 == 0:

##     print(f"{num} can be divided by 5")

##     break

##   print(f"{num} cannot be divided by 5")

##   num+=1

#' 
## num = 0

## while num<10:

##   num+=1

##   if num % 5 == 0:

##     continue

##   print(f"{num} cannot be divided by 5")

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
## title = "I love programming basics for Biostatistics!"

## title.find("I")

## title.find("love")

## title.find("o")

## title.find("o", 4) ## starting searching index is 4

## title.find("XX")

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
## title = "I love programming basics for Biostatistics!"

## "love" in title

## "computing" in title

## "XX" in title

## title.endswith("computing!")

## title.startswith("I love")

## title.count("l")

#' 
#' 
#' join
#' ===
#' 
## seq = ["1", "2", "3", "4", "5"]

## sep = "+"

## sep.join(seq)

## "".join(seq)

#' 
## dirs =( "", "usr", "bin", "env")

## "/".join(dirs)

## sep = "+"

## print("C:" + "\\".join(dirs)) ## single \ has special meaning: treating special symbol as regular symbol

#' 
#' split
#' ===
#' 
#' - reverse operator of join.
#' 
## longSeq = "1+2+3+4+5"

## longSeq.split("+")

## longSeq.split("3")

#' 
## "Using the default value".split()

#' 
#' 
#' lower, upper, title
#' ===
#' 
## sentence = "I like programming basics for Biostatistics!"

## sentence.lower()

## sentence.upper()

## sentence.title()

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
## a = "   internal   whitespace is kept     "

## a.strip()

## 

## b = "*** SPAM * for * everyone!!! ***"

## b.strip(" *!")

#' 
#' - also works for new line \n
#' 
## c = "\na\nb\n\n\nc\n\n"

## c.strip()

#' 
#' replace
#' ===
#' 
## a = "This is a cat!"

## a.replace("This", "That")

## a.replace("is", "eez")

#' 
#' 
#' 
#' 
#' 
#' file operation (read)
#' ===
#' 
#' https://caleb-huo.github.io/teaching/data/misc/my_file.txt
#' 
#' - open file, display, and close (release memory)
#' 
## file = open("my_file.txt")

## contents = file.read()

## print(contents)

## file.close()

#' 
#' - Alternative approach without closing step
#' 
## with open("my_file.txt") as file:

##     contents = file.read()

##     print(contents)

#' 
#' file operation (read)
#' ===
#' 
#' - readlines()
#'   - read multilple lines, 
#'   - save the result in a list
#'     - each element of the list contains a line
#' 
## myfile = "my_file.txt"

## with open(myfile) as file:

##     lines = file.readlines()

## 

## for aline in lines:

##     print(aline.strip())

#' 
#' 
#' file operation (write)
#' ===
#' 
#' - write to file (overwrite original file) 
## with open("new_file.txt", mode="w") as file:

##     file.write("I like python!")

#' 
#' - append to file (append at the end of the original file) 
## with open("new_file.txt", mode="a") as file:

##     file.write("We like python!")

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
## try:

##     file = open("a_file.txt")

##     print(1 + "2")

## except FileNotFoundError:

##     print("Catch FileNotFoundError")

## except TypeError as error_message:

##     print(f"Here is the error: {error_message}.")

## else:

##     content = file.read()

##     print(content)

## finally: ## will happen no matter what happens

##     file.close()

##     print("File was closed.")

#' 
#' datetime
#' ===
#' 
#' - The datetime module supplies classes for manipulating dates and times.
#' 
## import datetime as dt

## now = dt.date.today() ## date only

## now.year

## now.month

## now.day

## # now.weekday()

#' 
## birthday = dt.date(1995, 7, 31)

## age = now - birthday

## age.days

#' 
#' datetime
#' ===
#' 
## now = dt.datetime.now()

## now.year

## now.month

## now.day

## now.hour

#' 
#' --- 
#' 
## now.minute

## now.second

## now.microsecond

## now.weekday()

## 

#' 
#' datetime
#' ===
#' 
## now = dt.datetime.now()

## 

## print(f'{now:%Y-%m-%d %H:%M}')

#' 
#' 
#' more on python data structure
#' ===
#' 
#' - list
#' - dictionary
#' - tuple
#' 
#' 
#' 
#' list: creation and assignment
#' ===
#' 
#' - create a list from a string
#' 
## list("Hello")

#' 
#' - create a list directly
#' 
## x = [1, 1, 1]

## x

#' 
#' - change element
#' 
## x = [1, 2, 3]

## x[1] = 0

## x

#' 
#' list: deletion and slice assignment
#' ===
#' 
## names = ["Alice", "Beth", "Carl", "Dan", "Emily"]

## names

## del names[2]

## names

#' 
#' - slice assignment
#' 
## names = list("Lucas")

## names[3:] = list("ky")

## names

## "".join(names)

#' 
#' ---
#' 
#' - slice assignment can be unequal length
#' 
## names = list("Lucas")

## names[1:] = list("emonade")

## "".join(names)

#' 
#' - slice assignment can be used as insertion or deletion
#' 
## numbers = [1, 5]

## numbers[1:1] = [2, 3, 4]

## numbers

#' 
## numbers = list(range(1,6))

## numbers

## numbers[1:4] = []

## numbers

#' 
#' list: append and count
#' ===
#' 
#' - append
#' 
## alist = [0,1,2]

## alist.append(3)

## alist

#' 
#' - count
#' 
## asentence = "to be or not to be"

## alist = asentence.split()

## alist.count("to")

## 

## x = [[1,2], 1, 2, 1, [2, 1, [1,2]]]

## x.count(1)

## x.count([1,2])

#' 
#' 
#' list: extend
#' ===
#' 
#' - extend (recommended for efficiency and readibility)
#' 
## a = [0,1,2]; b = [3,4,5]

## a.extend(b)

## a

#' 
#' - direct +
#' 
## a = [0,1,2]; b = [3,4,5]

## a + b

## a

#' 
#' - slice assignment
#' 
## a = [0,1,2]; b = [3,4,5]

## a[len(a):] = b

## a

#' 
#' 
#' list: index
#' ===
#' 
#' - index: will return the first match
#' 
## asentence = "to be or not to be"

## alist = asentence.split()

## alist

## 

## alist.index("to")

## alist.index("not")

## alist[3]

#' 
#' ```
#' alist.index("XX")
#' ```
#' 
#' 
#' 
#' list: insert
#' ===
#' 
#' - insert
#' 
## alist = [1,2,3,5,6]

## alist.insert(3, "four")

## alist

#' 
#' - insert with slice assignment
#' 
## alist = [1,2,3,5,6]

## alist[3:3] = ["four"]

## alist

#' 
#' 
#' 
#' list: pop
#' ===
#' 
#' - pop: return the last element of a list
#'   - opposite of append
#' 
## x = list(range(10))

## x.pop()

## x

## x.pop()

## x

#' 
#' 
#' list: remove
#' ===
#' 
#' - remove
#' 
## asentence = "to be or not to be"

## alist = asentence.split()

## alist

## 

## alist.remove("to")

## alist

#' 
#' ```
#' alist.remove("XX")
#' ```
#' 
#' - compare pop and remove
#'   - remove has no return value, and remove the first appearance of certain value
#'   - pop has return value, and pop up the last element of a list
#' 
#' 
#' 
#' list: reverse and sort
#' ===
#' 
#' - reverse
#' 
## x = ["a", "b", "c"]

## x.reverse()

## x

#' 
#' - sort: sort method has no return value (in-place operator)
#' 
## x = [5, 3, 4]

## x.sort()

## x

#' 
## y = ["b", "c", "a"]

## y.sort()

## y

#' 
#' 
#' ---
#' 
#' - no return values:
#' 
## x = [5, 3, 4]

## y = x.sort()

## print(y)

#' 
#' - with return values:
#' 
## x = [5, 3, 4]

## y = sorted(x)

## print(y)

#' 
#' 
#' 
#' list: sort
#' ===
#' 
#' - sort
#' 
## x = [5, 3, 4]

## y = x ## x and y are pointing to the same list

## y.sort()

## 

## print(x)

## print(y)

#' 
#' 
## x = [5, 3, 4]

## y = x[:] ## y is a slice assignment of x, thus a new variable

## y.sort()

## 

## print(x)

## print(y)

#' 
#' references and values
#' ===
#' ![](../figure/sort_reference.png){width=90%}
#' 
#' 
#' list: sort
#' ===
#' 
#' - sort
#' 
#' 
## x = ["aaa", "bb", "cccc"]

## x.sort(key = len)

## x

#' 
## x = [5, 3, 4]

## x.sort(reverse = True)

## print(x)

#' 
#' 
#' dictionary: basic operator
#' ===
#' 
#' - basic operator
#' 
## phonebook = {"Alice": 2341,

##             "Beth": 4971,

##             "Carl": 9401

## }

## phonebook

#' 
#' 
## len(phonebook)

## phonebook["Beth"]

#' 
#' 
#' 
#' dictionary: update and delete
#' ===
#' 
#' - update and delete
#' 
## phonebook["Alice"] = 1358

## phonebook

## 

## adict = {"Alice": 9572}

## phonebook.update(adict)

## phonebook

## 

## del phonebook["Carl"]

## "Beth" in phonebook

#' 
#' 
#' 
#' dictionary: clear
#' ===
#' 
#' - clear
#' 
## d = {}

## d['name'] = "Amy"

## d['age'] = 24

## d

## 

## d.clear()

## d

#' 
#' why clear is useful
#' ===
#' 
#' 
## x = {}

## y = x

## x['key'] = 'value'

## y

## 

## x = {} ## now x points to a new value {}

## y ## y points to the original value {'key': 'value'}

#' 
#' 
## x = {}

## y = x

## x['key'] = 'value'

## y

## 

## x.clear() ## clear the value x points to

## y ## y still points to what x points to

#' 
#' 
#' references and values (part 2)
#' ===
#' ![](../figure/clear_reference.png){width=90%}
#' 
#' 
#' 
#' copy
#' ===
#' 
#' - shallow copy
#'     - only the reference address of the object is copied
#' 
#' 
## d = {}

## d['username'] = "admin"

## d['machines'] = ["foo", "bar"]

## d

## 

## c = d.copy()

## c['username'] = "Alex" ## c['username'] points to a new value

## print(c)

## print(d)

#' 
#' 
## c['machines'].remove("bar") ## references don't change, the underlying values are changed.

## print(c)

## print(d)

#' 
#' references and values (shallow copy)
#' ===
#' ![](../figure/shallowCopy.png){width=90%}
#' 
#' 
#' copy
#' ===
#' 
#' - deep copy:
#'     - will make a new copy of the values
#' 
## from copy import deepcopy

## 

## d = {}

## d['username'] = "admin"

## d['machines'] = ["foo", "bar"]

## d

## 

## c = d.copy()

## dc = deepcopy(d)

## d['machines'].remove("bar")

## print(c)

## print(dc)

#' 
#' 
#' references and values (deep copy)
#' ===
#' ![](../figure/deepcopy.png){width=90%}
#' 
#' 
#' dictionary initialization: fromkeys
#' ===
#' 
#' - create keys for an empty dictionary.
#' 
## {}.fromkeys(["name", "age"])

#' 
#' - create keys for a dictionary
#' 
## dict.fromkeys(["name", "age"])

#' 
#' - set default values
#' 
## dict.fromkeys(["name", "age"], "unknown")

#' 
#' dictionary: get
#' ===
#' 
#' - get method is more flexible
#' - get is the same as indexing by keys when the key exists
#' 
## d = {"name": "Amy", "age": 24}

## d["name"]

## d.get("name")

#' 
#' - get will return None when the key doesn't exist
#' 
#' ```
#' d["XX"]
#' d.get("XX")
#' d.get("XX", "No exist") ## set your own return value for get
#' ```
#' 
#' dictionary: items
#' ===
#' 
#' - items() return all items of the dictionary
#' 
#' 
## phonebook = {"Alice": 2341,

##             "Beth": 4971,

##             "Carl": 9401

## }

## phonebook

## phonebook.items() ## this is an iterable

## list(phonebook.items())

#' 
#' dictionary: loops
#' ===
#' 
#' - can be used for looping a dictionary
#' 
## it = phonebook.items()

## for key, value in it:

##     print(key +  "--> " + str(value))

#' 
#' - if you only want the value, not the keys
#' 
## it = phonebook.items()

## for _, value in it:

##     print(str(value))

#' 
#' ---
#' 
#' - use key to iterate a dictionary for a loop
#' 
## for key in phonebook:

##     print(key +  "--> " + str(phonebook[key]))

#' 
#' 
#' - use values() method
#' 
## phonebook.values() ## this is an iterable

## list(phonebook.values())

## for i in phonebook.values():

##     print(i)

#' 
#' 
#' 
#' 
#' dictionary: pop and popitem
#' ===
#' 
#' - pop
#' 
## phonebook = {"Alice": 2341,

##             "Beth": 4971,

##             "Carl": 9401

## }

## phonebook.pop("Alice")

## phonebook

#' 
#' 
#' - popitem(): pop up the last item
#' 
## phonebook = {"Alice": 2341,

##             "Beth": 4971,

##             "Carl": 9401

## }

## phonebook.popitem()

## phonebook

#' 
#' tuple: review basics
#' ===
#' 
## atuple = (0,1,2)

## atuple += (3,4,5)

## atuple

#' 
#' 
## btuple = (0, 1, 1, ['I', 'like',  'python'])

## btuple[3][0] = 'You'

## print(btuple)

## print(btuple.count(1))

## print(btuple.index(['You', "like", 'python']))

#' 
#' 
#' Reference
#' ===
#' 
#' - https://docs.python.org/3/index.html
#' - https://github.com/Apress/beginning-python-3ed
#' 
