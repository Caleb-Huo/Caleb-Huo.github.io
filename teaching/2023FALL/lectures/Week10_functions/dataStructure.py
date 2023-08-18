#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday Oct 26th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Python data structure"
#' ---
#' 

#' 
#' 
#' 
#' Outlines
#' ===
#' 
#' - list
#' - dictionary
#' - tuple
#' - set
#' 
#' 
#' list: creation and assignment
#' ===
#' 
#' - create a list from a string
#' 
list("Hello")
#' 
#' - create a list directly
#' 
x = [1, 1, 1]
x
#' 
#' - change element
#' 
x = [1, 2, 3]
x[1] = 0
x
#' 
#' list: deletion and slice assignment
#' ===
#' 
names = ["Alice", "Beth", "Carl", "Dan", "Emily"]
names
del names[2]
names
#' 
#' - slice assignment
#' 
names = list("Lucas")
names[3:] = list("ky")
names
"".join(names)
#' 
#' ---
#' 
#' - slice assignment can be unequal length
#' 
names = list("Lucas")
names[1:] = list("emonade")
"".join(names)
#' 
#' - slice assignment can be used as insertion or deletion
#' 
numbers = [1, 5]
numbers[1:1] = [2, 3, 4]
numbers
#' 
numbers = list(range(1,6))
numbers
numbers[1:4] = []
numbers
#' 
#' list: append and count
#' ===
#' 
#' - append
#' 
alist = [0,1,2]
alist.append(3)
alist
#' 
#' - count
#' 
asentence = "to be or not to be"
alist = asentence.split()
alist.count("to")

x = [[1,2], 1, 2, 1, [2, 1, [1,2]]]
x.count(1)
x.count([1,2])
#' 
#' 
#' list: extend
#' ===
#' 
#' - extend (recommended for efficiency and readibility)
#' 
a = [0,1,2]; b = [3,4,5]
a.extend(b)
a
#' 
#' - direct +
#' 
a = [0,1,2]; b = [3,4,5]
a + b
a
#' 
#' - slice assignment
#' 
a = [0,1,2]; b = [3,4,5]
a[len(a):] = b
a
#' 
#' 
#' list: index
#' ===
#' 
#' - index: will return the first match
#' 
asentence = "to be or not to be"
alist = asentence.split()
alist

alist.index("to")
alist.index("not")
alist[3]
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
alist = [1,2,3,5,6]
alist.insert(3, "four")
alist
#' 
#' - insert with slice assignment
#' 
alist = [1,2,3,5,6]
alist[3:3] = ["four"]
alist
#' 
#' 
#' 
#' list: pop
#' ===
#' 
#' - pop: return the last element of a list
#'   - opposite of append
#' 
x = list(range(10))
x.pop()
x
x.pop()
x
#' 
#' 
#' list: remove
#' ===
#' 
#' - remove
#' 
asentence = "to be or not to be"
alist = asentence.split()
alist

alist.remove("to")
alist
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
x = ["a", "b", "c"]
x.reverse()
x
#' 
#' - sort: sort method has no return value (in-place operator)
#' 
x = [5, 3, 4]
x.sort() 
x
#' 
y = ["b", "c", "a"]
y.sort()
y
#' 
#' 
#' ---
#' 
#' - don't do this:
#' 
x = [5, 3, 4]
y = x.sort() 
print(y)
#' 
#' - try this way
#' 
x = [5, 3, 4]
y = sorted(x) 
print(y)
#' 
#' 
#' 
#' list: sort
#' ===
#' 
#' - sort
#' 
x = [5, 3, 4]
y = x ## x and y are pointing to the same list
y.sort() 

print(x)
print(y)
#' 
#' 
x = [5, 3, 4]
y = x[:] ## y is a slice assignment of x, thus a new variable
y.sort() 

print(x)
print(y)
#' 
#' 
#' 
#' list: sort
#' ===
#' 
#' - sort
#' 
#' 
x = ["aaa", "bb", "cccc"]
x.sort(key = len) 
x
#' 
x = [5, 3, 4]
x.sort(reverse = True) 
print(x)
#' 
#' 
#' dictionary: basic operator
#' ===
#' 
#' - basic operator
#' 
phonebook = {"Alice": 2341, 
            "Beth": 4971,
            "Carl": 9401
}
phonebook
#' 
#' 
len(phonebook)
phonebook["Beth"]
#' 
#' 
#' 
#' dictionary: update and delete
#' ===
#' 
#' - update and delete
#' 
phonebook["Alice"] = 1358
phonebook

adict = {"Alice": 9572}
phonebook.update(adict)
phonebook

del phonebook["Carl"]
"Beth" in phonebook
#' 
#' 
#' 
#' dictionary: clear
#' ===
#' 
#' - clear
#' 
d = {}
d['name'] = "Amy"
d['age'] = 24
d

d.clear()
d
#' 
#' why clear is useful
#' ===
#' 
#' 
x = {}
y = x
x['key'] = 'value'
y

x = {} ## associate x with an empty dictionary
y ## y points to the original dictionary
#' 
#' 
x = {}
y = x
x['key'] = 'value'
y

x.clear() ## clear the original dictionary
y ## y still points to the original dictionary
#' 
#' 
#' copy
#' ===
#' 
#' - shallow copy
#'     - only the reference address of the object is copied
#' 
#' 
d = {}
d['username'] = "admin"
d['machines'] = ["foo", "bar"]
d

c = d.copy()
c['username'] = "Alex" ## replacement will not change the original dictionary
print(c)
print(d)
#' 
#' 
c['machines'].remove("bar") ## modification will change the original dictionary
print(c)
print(d)
#' 
#' copy
#' ===
#' 
#' - deep copy:
#'     - will make a new copy of everything.
#' 
from copy import deepcopy

d = {}
d['username'] = "admin"
d['machines'] = ["foo", "bar"]
d

c = d.copy()
dc = deepcopy(d)
d['machines'].remove("bar") 
print(c)
print(dc)
#' 
#' dictionary initialization: fromkeys
#' ===
#' 
#' - create keys for an empty dictionary.
#' 
{}.fromkeys(["name", "age"])
#' 
#' - create keys for a dictionary
#' 
dict.fromkeys(["name", "age"])
#' 
#' - set default values
#' 
dict.fromkeys(["name", "age"], "unknown")
#' 
#' dictionary: get
#' ===
#' 
#' - get method is more flexible
#' - get is the same as indexing by keys when the key exists
#' 
d = {"name": "Amy", "age": 24}
d["name"]
d.get("name")
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
phonebook = {"Alice": 2341, 
            "Beth": 4971,
            "Carl": 9401
}
phonebook
phonebook.items()
list(phonebook.items())
#' 
#' dictionary: loops
#' ===
#' 
#' - can be used for looping a dictionary
#' 
it = phonebook.items()
for key, value in it:
    print(key +  "--> " + str(value))
#' 
#' - if you only want the value, not the keys
#' 
it = phonebook.items()
for _, value in it:
    print(str(value))
#' 
#' ---
#' 
#' - use key to iterate a dictionary for a loop
#' 
for key in phonebook:
    print(key +  "--> " + str(phonebook[key]))
#' 
#' 
#' - use values() method
#' 
phonebook.values()
list(phonebook.values())
for i in phonebook.values():
    print(i)
#' 
#' 
#' dictionary: list comprehension
#' ===
names = ["Amy", "Beth", "Carl", "Dan", "Emily", "Frank"]

import random

students_scores = {name: random.randint(0, 100) for name in names}

passed_students = {key: value for (key, value) in students_scores.items() if value > 60}

passed_students
#' 
#' 
#' dictionary: pop and popitem
#' ===
#' 
#' - pop
#' 
phonebook = {"Alice": 2341, 
            "Beth": 4971,
            "Carl": 9401
}
phonebook.pop("Alice")
phonebook
#' 
#' 
#' - popitem(): pop up the last item
#' 
phonebook = {"Alice": 2341, 
            "Beth": 4971,
            "Carl": 9401
}
phonebook.popitem()
phonebook
#' 
#' tuple: review basics
#' ===
#' 
atuple = (0,1,2)
atuple += (3,4,5)
atuple
#' 
#' 
btuple = (0, 1, 1, ['I', 'like',  'python'])
btuple[3][0] = 'You'
print(btuple)
print(btuple.count(1))
print(btuple.index(['You', "like", 'python']))
#' 
#' set: create and add
#' ===
#' 
#' - sets are a collection of unordered unique elements
#' 
#' - create
#' 
this_set = {1, 1, 2, 3, 3, 3, 4} #create set
print(this_set)
#' 
#' - add
#' 
this_set = {0, 1, 2} #create set
this_set.add(3)
this_set
#' 
#' set: operators
#' ===
#' 
set_a = {"a", "b", "c", "d"}
set_b = {"c", "d", "e", "f"}
print(set_a.union(set_b), '----------', set_a | set_b)
print(set_a.intersection(set_b), '----------', set_a & set_b)
print(set_a.difference(set_b), '----------', set_a - set_b)
print(set_a.symmetric_difference(set_b), '----------', set_a ^ set_b)
set_a.clear()
set_a
#' 
#' 
#' Reference
#' ===
#' 
#' 
#' - https://docs.python.org/3/index.html
#' - https://github.com/Apress/beginning-python-3ed
#' 
