#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Tuesday Oct 24th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Objective Oriented Programming"
#' ---
#' 
## ----setup, include=FALSE----------------------------------------------------
library(reticulate)
use_python("/usr/local/bin/python3.11")

#' 
#' 
#' 
#' 
#' Introduction
#' ===
#' 
#' - Procedural Programming: typically a list of instructions that execute line by line 
#'     - Advantage: 
#'         - general-purpose programming
#'         - simple, intutive and straightforward
#'         - easy to keep track of the program flow
#'     - Disadvantage:
#'         - the programming code is usually not reusable
#'         - not scalale for large applications
#' 
#' Introduction
#' ===
#' 
#' - Objective Oriented Programming (OOP): to bundle variables and methods into object
#'     - Advantage: 
#'         - reusable
#'         - good to capture the complex relationship among different procedures.
#'         - readable, easy to maintain
#'         - hide the implementation from the user's end
#'     - Disadvantage:
#'         - Usually larger size
#' 
#' 
#' 
#' A toy example
#' ===
#' 
## class Person:

## 

##     def setName(self, aname):

##         self.name = aname

## 

##     def greet(self):

##         print("Hi there, my name is " + self.name)

## 

## a = Person()

## a.setName("Lucas")

## a.greet()

## 

#' 
#' - Class: Person (usually, the first letter of a class is capitalized)
#' - Object: a: a realization (object) of the class
#' - Attribute: name
#' - Method: setName(); greet()
#' 
#' 
#' OOP vocabulary
#' ===
#' 
#' - Class
#'     - a blueprint which is consisting of methods and attributes.
#' - Object
#'     - an instance of a class. It can help to think of objects as something in the real world like  
#'         - a small dog (if class is animal)
#'         - a yellow shoe (if class is Dressing and Wearing) etc. 
#'         - objects can be more abstract.
#' - Attribute
#'     - a descriptor or characteristic. Examples would be color, length, size, etc. These attributes can take on specific values like blue, 3 inches, large, etc.
#' - Method
#'     - an action that a class or object could take.
#' 
#' 
#' multiple objects from the same class
#' ===
#' 
## a = Person()

## a.setName("Lucas")

## a.greet()

## 

## b = Person()

## b.setName("Amy")

## b.greet()

## 

## c = Person()

## c.setName("Beth")

## c.greet()

#' 
#' 
#' Class methods
#' ===
#' 
#' - when constructing a method **inside a class**, the first argument should be self
#'   - self refer to the class itself
#'   - with self, all its attributes will be accessible **within the class**
#' 
#' - when **an object** calls the method, we don't need the self argument
#' 
#' 
#' 
#' 
#' In class exercise (1)
#' ===
#' 
#' - construct a class Participant
#' - attribute:
#'     - name 
#'     - sex 
#'     - age
#' - method:
#'     - display_info(): 
#' 
#' ```
#' Create a new participant Amy, who is female and 49 years old.
#' When we use the display_info(), we expect to see
#' 
#' "Participant Amy, 49 yrs, female".
#'     
#' ```
#' 
#' ---
#' 
#' 
## class Participant:

## 

##     def setName(self,name):

##         self.name = name

## 

##     def setAge(self,age):

##         self.age = age

## 

##     def setGender(self,sex):

##         self.sex = sex

## 

##     def display_info(self):

##         print(f"Participant {self.name}, {self.age} yrs, {self.sex}")

## 

## aParticipant = Participant()

## aParticipant.setName("Amy")

## aParticipant.setAge(49)

## aParticipant.setGender("Sex")

## 

## aParticipant.display_info()

#' 
#' 
#' 
#' Attributes initialization
#' ===
#' 
## class Person:

## 

##     def __init__(self):

##         self.name = "John"

## 

##     def setName(self, aname):

##         self.name = aname

## 

##     def greet(self):

##         print("Hi there, my name is " + self.name)

## 

## a = Person()

## a.setName("Lucas")

## a.greet()

## 

## b = Person()

## b.greet()

#' 
#' 
#' Attributes initialization with a user's input
#' ===
#' 
#' - this allows to set attributes when constructing an object
#' 
## class Person:

## 

##     def __init__(self, aname):

##         self.name = aname

## 

##     def setName(self, aname):

##         self.name = aname

## 

##     def greet(self):

##         print("Hi there, my name is " + self.name)

## 

## a = Person("Lucas")

## b = Person("Amy")

## 

## a.greet()

## b.greet()

#' 
#' Default value in the initialization
#' ===
#' 
## class Person:

## 

##     def __init__(self, aname="John"):

##         self.name = aname

## 

##     def setName(self, aname):

##         self.name = aname

## 

##     def greet(self):

##         print("Hi there, my name is " + self.name)

## 

## a = Person("Lucas")

## a.greet()

## b = Person()

## b.greet()

#' 
#' 
#' 
#' Class methods
#' ===
#' 
#' - when constructing a method **inside a class**, the first argument should be self
#'   - self refer to the class itself
#'   - with self, all its attributes will be accessible **within the class**
#'   
#' - to initialize the attributes of a class
#'   - need the \_\_init\_\_() method
#'   - \_\_init\_\_() also needs self as its argument
#' 
#' - when **an object** calls the method, we don't need the self argument
#' 
#' 
#' In class exercise (2)
#' ===
#' 
#' - construct a class Participant
#' - attribute:
#'     - name
#'     - sex
#'     - age
#' - *initalize these attributes when constructing the object*
#' - method:
#'     - display_info(): 
#' 
#' ```
#' Create a new participant Amy, who is female and 49 years old.
#' When we use the display_info(), we expect to see
#' 
#' "Participant Amy, 49 yrs, female".
#'     
#' ```
#' 
#' ---
#' 
## class Participant:

##     def __init__(self, name, age, sex):

##         self.name = name

##         self.age = age

##         self.sex = sex

## 

##     def setName(self,name):

##         self.name = name

## 

##     def setAge(self,age):

##         self.age = age

## 

##     def setGender(self,sex):

##         self.sex = sex

## 

##     def display_info(self):

##         print(f"Participant {self.name}, {self.age} yrs, {self.sex}")

## 

## aParticipant = Participant(name="Amy", age=49, sex="female")

## aParticipant.display_info()

#' 
#' 
#' 
#' 
#' private method
#' ===
#' 
#' - to set a method to be private, use could use __method
#' - private method 
#'     - only can be used within the class
#'     - if an object is created, private method is not accessible
#' 
## class Secretive:

##     def __inaccessible(self):

##         print("Bet you can't see me...")

##     def accessible(self):

##         print("The secret message is:")

##         self.__inaccessible()

## 

## 

## s = Secretive()

## #s.__inaccessible() ## does not work

## s.accessible()

#' 
#' 
#' Class attributes
#' ===
#' 
## class Person:

## 

##     def __init__(self, aname):

##         self.name = aname

## 

##     def getName(self):

##         return(self.name)

## 

## a = Person("Lucas")

#' 
#' 
#' - We can define methods to retrieve attributes.
#' 
## a.getName()

#' 
#' - After construct an object (i.e., obj), we can also use obj.attribute to access the object.
#' 
## a.name  ## alternative way to get attribute

#' 
#' 
#' 
#' private attribute
#' ===
#' 
#' - to set an attribute to be private, use could use __attribute
#' 
## class Person:

## 

##     def __init__(self, aname):

##         self.__name = aname

## 

##     def getName(self):

##         return(self.__name)

## 

## a = Person("Lucas")

## a.getName()

## #a.__name ## does not work

#' 
#' 
#' In class exercise (3)
#' ===
#' 
#' - construct a class Participant
#' - attribute:
#'     - name
#'     - sex
#'     - age
#' - initalize these attributes when constructing the object*
#' - private attribute:
#'   - *phone_number (private attribute)*
#' - method:
#'     - display_info(): 
#'     - show_contact_number(): 
#' 
#' 
#' ```
#' Create a new participant Amy, who is female and 49 years old.
#' Also create her phone_number (123-456-7890) by using setPhoneNumber() method.
#' The phone_number is not directly accessible via class.phone_number.
#' It is accessible via class.show_contact_number()
#' ```
#'             
#' ---
#' 
## class Participant:

##     def __init__(self, name, age, sex):

##         self.name = name

##         self.age = age

##         self.sex = sex

## 

##     def setPhoneNumber(self,anumber):

##         self.__phone_number = anumber

## 

##     def display_info(self):

##         print(f"Participant {self.name}, {self.age} yrs, {self.sex}")

## 

##     def show_contact_number(self):

##         print(self.__phone_number)

## 

## aParticipant = Participant(name="Amy", age=49, sex="female")

## aParticipant.setPhoneNumber("123-456-7890")

## aParticipant.show_contact_number()

#' 
#' 
#' 
#' extra argument and keywords in the class initialization
#' ===
#' 
## class Car:

## 

##     def __init__(self, **kw):

##         self.make = kw["make"]

##         # self.model = kw["model"]

##         self.model = kw.get("model")

## 

## my_car = Car(make = "Chevo", model = "Malibu")

## print(my_car.model)

## 

## my_car = Car(make = "Chevo")

## print(my_car.model)

#' 
#' 
#' a Class with attributes being other class's objects
#' ===
#' 
#' - previous Person class
#' 
## class Person:

## 

##     def __init__(self, aname="John"):

##         self.name = aname

## 

##     def setName(self, aname):

##         self.name = aname

## 

##     def getName(self):

##         return(self.name)

## 

##     def greet(self):

##         print("Hi there, my name is " + self.name)

#' 
#' - Goal: create a Group class, which contain a group of persons.
#' 
#' a Group class
#' ===
#' 
## class Group:

## 

##     def __init__(self):

##         self.persons = []

## 

##     def add(self, aperson):

##         self.persons.append(aperson)

## 

##     def getNameAll(self):

##         res = [aperson.getName() for aperson in self.persons]

##         return(res)

## 

## agroup = Group()

## 

## agroup.add(Person("Amy"))

## agroup.add(Person("Beth"))

## agroup.add(Person("Carl"))

## 

## agroup.getNameAll()

#' 
#' 
#' 
#' 
#' In class exercise (4)
#' ===
#' 
#' - construct a class Participant
#'     - attribute:
#'         - name
#'         - sex
#'         - age
#'     - initalize these attributes when constructing the object*
#'     - method:
#'         - display_info(): 
#' 
#'     
#' - construct a class Study
#'     - attribute:
#'         - participants: a list of individual participants
#'     - methods:
#'         - get_mean_age()
#'         - get_percent_female()
#'         - display the info for all participants (one in each line)
#' 
#' 
#' ```
#' Create a study with 5 participants, Amy, Beth, Carl, Dan, Emily.
#' Sex are randomly choose from male and female.
#' Age are randomly selected from 20-60.
#' 
#' And then show the mean age and percentage of females.
#' Also show the info for all participants
#' ```
#' 
#'             
#' ---
#' 
#' 
## class Participant:

## 

##     def __init__(self, name, age, sex):

##         self.name = name

##         self.age = age

##         self.sex = sex

## 

## 

##     def display_info(self):

##         print(f"Participant {self.name}, {self.age} yrs, {self.sex}")

## 

#' 
#' ---
#' 
#' 
## class Study:

## 

##     def __init__(self):

##         self.participants = []

## 

##     def enroll(self, aparticipant):

##         self.participants.append(aparticipant)

## 

##     def get_mean_age(self):

##         pass

## 

##     def get_percent_female(self):

##         pass

## 

##     def display_all(self):

##         for aparticipant in self.participants:

##             aparticipant.display_info()

## 

## 

#' 
#' ---
#' 
#' 
## astudy = Study()

## names = ["Amy", "Beth", "Carl", "Dan", "Emily"]

## genders = ["male", "female"]

## import random

## for aname in names:

##   aage = random.randint(20,80)

##   asex = random.choice(genders)

##   aparticipant = Participant(aname, aage, asex)

##   astudy.enroll(aparticipant)

## 

## astudy.get_mean_age()

## astudy.get_percent_female()

## astudy.display_all()

#' 
#' 
#' Class inheritance
#' ===
#' 
#' - Inheritance allows us to define a class that inherits all the methods and properties from another class.
#'     - Parent class is the class being inherited from
#'     - Child class is the class that inherits from another class
#' 
#' - to inherit a class, using the parental class as the argument for the child class
#' 
#' ```
#' class Child(Parent):
#' ```
#' 
#' ---
#' 
#' - parent class
#' 
## class Person:

## 

##     def __init__(self):

##         self.name = "John"

## 

##     def setName(self, aname):

##         self.name = aname

## 

##     def greet(self):

##         print("Hi there, my name is " + self.name)

#' 
#' - child class
#' 
## class Student(Person):

##     pass

#' 
## astudent = Student()

## astudent.greet()

#' 
#' Class inheritance
#' ===
#' 
#' The child class can have additional attributes and methods compared to the parent class
#' 
## class Person:

## 

##     def __init__(self):

##         self.name = "John"

## 

##     def setName(self, aname):

##         self.name = aname

## 

##     def greet(self):

##         print("Hi there, my name is " + self.name)

#' 
#' ---
#' 
#' - child class
#'     - it inherits the parental attributes.
#' 
## class Student(Person):

## 

##     def __init__(self):

##         self.courses = {}

## 

##     def add_course(self, course, grade):

##         self.courses.update({course:grade})

## 

##     def get_ave_grade(self):

##         ave_grades = sum(list(self.courses.values()))/len(self.courses)

##         return(ave_grades)

## 

#' 
#' ---
#' 
## astudent = Student()

## astudent.setName("Levi")

## astudent.greet()

## 

## astudent.add_course("math", 4.0)

## astudent.add_course("physics", 3.8)

## astudent.add_course("history", 3.7)

## astudent.get_ave_grade()

#' 
#' 
## issubclass(Person, Student)

## issubclass(Student, Person)

#' 
#' 
#' 
#' Super method for python class
#' ===
#' 
#' - super() allows to access methods of the parental class.
#' 
## class Animal():

##   def __init__(self, animal_type):

##     print('Animal Type:', animal_type)

## 

## class Dog(Animal):

##   def __init__(self):

## 

##     # call superclass

##     super().__init__('dog')

##     ## super(Dog, self).__init__('dog') ## alternatively

##     print('Wolffff')

## 

## adog = Dog()

## 

#' 
#' 
#' 
#' 
#' In class exercise (5)
#' ===
#' 
#' - previous exercise (2)
#' 
## class Participant:

##     def __init__(self, name, age, sex):

##         self.name = name

##         self.age = age

##         self.sex = sex

## 

##     def display_info(self):

##         print(f"Participant {self.name}, {self.age} yrs, {self.sex}")

## 

## aParticipant = Participant(name="Amy", age=49, sex="female")

## aParticipant.display_info()

#' 
#' ---
#' 
#' - assume the Participant class (in exercise 2) is the parent class, let's define a child class (i.e., Athlete) with an additional attribute and method.
#'     - attribute: strength (set strength=100 in initialization)
#'     - method: train (will increase strength by 1)
#' 
#' - pseudocode for the train method
#' 
#' ```
#' def train(self):
#'     strength += 1
#'     print(my strength is {strength})
#' ```
#' 
#' 
#' Athlete Class
#' ===
#' 
## class Athlete(Participant):

## 

##     def __init__(self, name, age, sex):

##         super(Athlete, self).__init__(name, age, sex) ## need this to initialize the parental class

##         ## alternatively, super().__init__(name, age, sex)

##         self.strength = 100

## 

##     def train(self):

##         self.strength += 1

##         print(f"my strength is {self.strength}")

## 

#' 
#' ---
#' 
## aAthlete = Athlete(name="Amy", age=49, sex="female")

## aAthlete.display_info()

## aAthlete.strength

## aAthlete.train()

## aAthlete.strength

#' 
#' 
#' Multiple inheritance
#' ===
#' 
#' - inherit the attribtues and methods from multiple classes
#' 
## class Calculator:

##     def calculate(self, expression):

##         self.value = eval(expression) ## eval is to evaluate some expression in python

## 

## class Talker:

##     def talk(self):

##         print("Hi, my value is", self.value)

## 

## class TalkingCalculator(Calculator, Talker):

##     pass

#' 
## a = TalkingCalculator()

## a.calculate("1+1")

## a.talk()

#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' 
#' Exercise (will be part of HW4)
#' ===
#' 
#' - Create a Student class
#'   - attributes:
#'       - Name
#'       - Hobby
#'       - Year_in_colledge
#'       - GPA
#'       - study_time (per week)
#'   - methods:
#'       - introduction() 
#'           - Hello, this is **Dan**. I am a **freshman**. My hobby is **Football**!
#'       - getGPA()
#'       - getName()
#'       - update_GPA()
#'           - The new GPA = old GPA + log10(study_time) - 1. 
#'           - Note that the new GPA should not be above 4.0. If it is above, just set it to be 4.0.
#'       
#' 
#' Exercise (will be part of HW4)
#' ===
#' 
#' - read in file
#'   - The targetted file is located here: https://caleb-huo.github.io/teaching/data/Python/Student_data.csv. 
#'   - Read in this file and skip the header line ('Name', 'Hobby', 'Year_in_colledge', 'Initial_GPA', 'Study_time'). 
#'   - Prepare the data in the format of a list, with each element of the list being a tuple. 
#'   - The first element of the list should be ('Dan', 'Football', 'freshman', '3.1', '10'). 
#'   - Print out the result.
#' 
#' - construct a student object using Dan's information
#' 
#' 
#' Exercise (will be part of HW4)
#' ===
#' 
#' - Create a StudyGroup class
#'   - attributes:
#'       - students: a list of student objects
#'   - methods:
#'       - add() 
#'           - add a student object to the **students list**
#'       - get_roster()
#'           - print out all students' names as a list.
#'       - introduction_all()
#'           - print out the introduction of each student line by line.
#'           - The order of the introduction should be the alphabetic order. 
#'           - For example, the first introduction should come from student Amy. 
#'       - get_high_GPA()
#'           - in the format of dictionary {name:GPA}
#'       - get_ave_GPA()
#'       - group_study()
#'           - This method will let each student in the class study according to their study time. 
#' 
#' 
#' Exercise (will be part of HW4)
#' ===
#' 
#' - Create the StudyGroup object, 
#' - add all 10 students to the StudyGroup object, 
#' - print out all students' names as a list. 
#' - introduce all students in alphabetic order
#' - get average GPA
#' - get average GPA after group study
#' - get highest GPA in the format of dictionary {name:GPA}
#' 
#' 
#' Reference
#' ===
#' 
#' - https://docs.python.org/3/index.html
#' - https://realpython.com/python3-object-oriented-programming/
#' - https://towardsdatascience.com/python-procedural-or-object-oriented-programming-42c66a008676
#' - https://github.com/Apress/beginning-python-3ed
