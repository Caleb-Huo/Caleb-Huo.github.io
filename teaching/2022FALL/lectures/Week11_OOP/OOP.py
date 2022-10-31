#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Oct 31st, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Objective Oriented Programming"
#' ---
#' 

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
class Person:

    def setName(self, aname):
        self.name = aname
    
    def greet(self):
        print("Hi there, my name is " + self.name)

a = Person()
a.setName("Lucas")
a.greet()

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
a = Person()
a.setName("Lucas")
a.greet()

b = Person()
b.setName("Amy")
b.greet()

c = Person()
c.setName("Beth")
c.greet()
#' 
#' Attributes initialization
#' ===
#' 
class Person:
    
    def __init__(self):
        self.name = "John"
    
    def setName(self, aname):
        self.name = aname
    
    def greet(self):
        print("Hi there, my name is " + self.name)

a = Person()
a.setName("Lucas")
a.greet()

b = Person()
b.greet()
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
#' In class exercise (1)
#' ===
#' 
#' - construct a class Bird
#' - attribute:
#'     - hungry: set to be true in the initialization
#' - method:
#'     - eat:
#' 
#' ```
#' pseudocode for the eat method:
#'     if hungry==True:
#'         print("Aaah")
#'         hungry=False
#'     else:
#'         print("No thanks")
#' ```
#' 
#' The Bird class (1)
#' ===
#' 
class Bird:

    def __init__(self):
        self.hungry = True
    
    def eat(self):
        if self.hungry:
            print("Aaah")
            self.hungry = False
        else:
            print("No, thanks")    

aBird = Bird()
aBird.eat()
aBird.eat()
#' 
#' 
#' 
#' Attributes initialization with a user's input
#' ===
#' 
#' - this allows to set attributes when constructing an object
#' 
class Person:
    
    def __init__(self, aname):
        self.name = aname
    
    def setName(self, aname):
        self.name = aname
    
    def greet(self):
        print("Hi there, my name is " + self.name)

a = Person("Lucas")
b = Person("Amy")

a.greet()
b.greet()
#' 
#' Default value in the initialization
#' ===
#' 
class Person:
    
    def __init__(self, aname="John"):
        self.name = aname
    
    def setName(self, aname):
        self.name = aname
    
    def greet(self):
        print("Hi there, my name is " + self.name)
    
a = Person("Lucas")
a.greet()
b = Person()
b.greet()
#' 
#' 
#' Class method (what if there is no self)
#' ===
#' 
#' 
class Bird0:
    song = 'Squaawk'
    def sing(self):
        print(self.song)
    def sing2():
        print(song)


bird = Bird0()
bird.sing()
#bird.sing2()
#' 
#' private method
#' ===
#' 
#' - to set a method to be private, use could use __method
#' - private method 
#'     - only can be used within the class
#'     - if an object is created, private method is not accessible
#' 
class Secretive:
    def __inaccessible(self):
        print("Bet you can't see me...")
    def accessible(self):
        print("The secret message is:")
        self.__inaccessible()


s = Secretive()
#s.__inaccessible() ## does not work
s.accessible()
#' 
#' 
#' Class attributes
#' ===
#' 
class Person:
    
    def __init__(self, aname):
        self.name = aname
    
    def getName(self):
        return(self.name)
    
a = Person("Lucas")
#' 
#' 
#' - We can define methods to retrieve attributes.
#' 
a.getName()
#' 
#' - After construct an object (i.e., obj), we can also use obj.attribute to access the object.
#' 
a.name  ## alternative way to get attribute
#' 
#' 
#' 
#' private attribute
#' ===
#' 
#' - to set an attribute to be private, use could use __attribute
#' 
class Person:
    
    def __init__(self, aname):
        self.__name = aname
    
    def getName(self):
        return(self.__name)
    
a = Person("Lucas")
a.getName()
#a.__name ## does not work
#' 
#' 
#' In class exercise (2)
#' ===
#' 
#' - construct a class Bird2
#' - attribute:
#'     - color: string
#'     - hungry: Boolean type
#' - method:
#'     - getColor
#'     - eat:
#' - initialization:
#'     - hungry default is True
#'     - color: default is "yellow"
#'     - users are allowed to choose other arguments
#' ```
#' pseudocode for the eat method:
#'     if hungry==True:
#'         print("Aaah")
#'         hungry=False
#'     else:
#'         print("No thanks")
#' ```
#'             
#' The Bird class (2)
#' ===
#' 
class Bird2:

    def __init__(self, color="yellow", hungry=True):
        self.color = color
        self.hungry = hungry
    
    def getColor(self):
        return(self.color)
    
    def eat(self):
        if self.hungry:
            print("Aaah")
            self.hungry = False
        else:
            print("No, thanks")    
#' 
#' ---
#' 
bBird = Bird2("red")
bBird.getColor()
bBird.eat()
bBird.eat()
#' 
#' a Class with attributes being other class's objects
#' ===
#' 
#' - previous Person class
#' 
class Person:
    
    def __init__(self, aname="John"):
        self.name = aname
    
    def setName(self, aname):
        self.name = aname
    
    def getName(self):
        return(self.name)
    
    def greet(self):
        print("Hi there, my name is " + self.name)
#' 
#' - Goal: create a Group class, which contain a group of persons.
#' 
#' a Group class
#' ===
#' 
class Group:
    
    def __init__(self):
        self.persons = []
    
    def add(self, aperson):
        self.persons.append(aperson)
    
    def getNameAll(self):
        res = [aperson.getName() for aperson in self.persons]
        return(res)
        
agroup = Group()

agroup.add(Person("Amy"))
agroup.add(Person("Beth"))
agroup.add(Person("Carl"))

agroup.getNameAll()
#' 
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
class Person:
    
    def __init__(self):
        self.name = "John"
    
    def setName(self, aname):
        self.name = aname
    
    def greet(self):
        print("Hi there, my name is " + self.name)
#' 
#' - child class
#' 
class Student(Person):
    pass
#' 
astudent = Student()
astudent.greet()
#' 
#' Class inheritance
#' ===
#' 
#' The child class can have additional attributes and methods compared to the parent class
#' 
#' - parent class
#' 
class Filter:
    def __init__(self):
        self.blocked = [0, 1]
    def filter(self, sequence):
        return [x for x in sequence if x not in self.blocked]
#' 
f = Filter()
f.filter([1,2,3])
#' 
#' ---
#' 
#' - child class
#'     - it inherits the methods.
#' 
class SPAMFilter(Filter):
    def __init__(self):
        self.blocked = ['SPAM']
#' 
s = SPAMFilter()
s.filter(["SPAM","bacan"])
#' 
issubclass(SPAMFilter, Filter)
issubclass(Filter, SPAMFilter)
#' 
#' 
#' ---
#' 
#' - child class
#'     - it inherits the parental attributes.
#' 
class SPAMFilter2(Filter):
    pass
#' 
s2 = SPAMFilter2()
s2.filter([1, 2, 3])
#' 
#' - child class
#'     - but cannot modify the parental attributes
#' 
#' ```
#' class SPAMFilter3(Filter):
#'     def __init__(self):
#'         self.blocked.append('SPAM')
#' ```
#' 
#' ```
#' s3 = SPAMFilter3()
#' s3.filter([1, 2, "SPAM"])
#' ```
#' 
#' 
#' 
#' Super method for python class
#' ===
#' 
#' - super() allows to access methods of the parental class.
#' 
class Animal(object):
  def __init__(self, animal_type):
    print('Animal Type:', animal_type)
    
class Mammal(Animal):
  def __init__(self):

    # call superclass
    super().__init__('Mammal')
    ## super(Mammal, self).__init__('Mammal') ## alternatively
    print('Mammals give birth directly')
    
dog = Mammal()

#' 
#' 
#' 
#' Super method for python class
#' ===
#' 
class Filter:
    def __init__(self):
        self.blocked = [0, 1]
    def filter(self, sequence):
        return [x for x in sequence if x not in self.blocked]
#' 
class SPAMFilter4(Filter):
    def __init__(self):
        super(SPAMFilter4, self).__init__()
        self.blocked.append('SPAM')
#' 
s4 = SPAMFilter4()
s4.filter([1, 2, "SPAM"])
#' 
#' 
#' In class exercise (3)
#' ===
#' 
#' - assume the Bird class (in exercise 1) is the parent class, let's define a child class (i.e., SongBird) with an additional attribute and method.
#'     - attribute: sound (set sound="tweet" in initialization)
#'     - method: sing
#' 
#' - pseudocode for the sing method
#' 
#' ```
#' def sing(self):
#'     print(self.sound * 3)
#' ```
#' 
#' 
#' SongBird Class
#' ===
#' 
class Bird:
    
    def __init__(self):
        self.hungry = True
    
    def eat(self):
        if self.hungry:
            print("Aaah")
            self.hungry = False
        else:
            print("No, thanks")    

class SongBird(Bird):
    
    def __init__(self):
        super(SongBird, self).__init__() ## need this to initialize the parental class
        self.sound = "tweet"
    
    def sing(self):
        print(self.sound * 3)

#' 
#' ---
#' 
aSongBird = SongBird()
aSongBird.hungry
aSongBird.sound
aSongBird.sing()
aSongBird.eat()
aSongBird.hungry
aSongBird.eat()
#' 
#' 
#' Multiple inheritance
#' ===
#' 
#' - inherit the attribtues and methods from multiple classes
#' 
class Calculator:
    def calculate(self, expression):
        self.value = eval(expression) ## eval is to evaluate some expression in python

class Talker:
    def talk(self):
        print("Hi, my value is", self.value)

class TalkingCalculator(Calculator, Talker):
    pass
#' 
a = TalkingCalculator()
a.calculate("1+1")
a.talk()
#' 
#' 
#' In class exercise (4)
#' ===
#' 
#' - assume the Bird2 class (in exercise 2) is the parent class, let's define a child class  (i.e., SongBird2) with an additional attribute and method.
#'     - attribute: sound (with default value TWEET)
#'     - method: sing
#'     - also allow to specify color and hungry when constructing the object. E.g.,
#'     ```
#'     SongBird2(color="yellow", hungry=True, sound = "TWEET")
#'     ```
#'     
#'     
#' - pseudocode for the sing method
#' 
#' ```
#' def sing(self):
#'     print(self.sound * 3)
#' ```
#' 
#' SongBird Class
#' ===
#' 
class Bird2:

    def __init__(self, color="yellow", hungry=True):
        self.color = color
        self.hungry = hungry
    
    def getColor(self):
        return(self.color)
    
    def eat(self):
        if self.hungry:
            print("Aaah")
            self.hungry = False
        else:
            print("No, thanks")    

## bBird = Bird2("red")
## bBird.getColor()
## bBird.eat()
## bBird.eat()
#' 
#' ---
#' 
class SongBird2(Bird2):
    
    def __init__(self, color="yellow", hungry=True, sound="TWEET"):
        super(SongBird2, self).__init__(color=color, hungry=hungry)
        self.sound = sound
    
    def sing(self):
        print(self.sound * 3)
#' 
#' ---
#' 
bSongBird = SongBird2(color="red")
bSongBird.sing()
print(bSongBird.getColor())
bSongBird.eat()
bSongBird.eat()
#' 
#' 
#' verify inheritance
#' ===
#' 
issubclass(Bird, SongBird)
issubclass(SongBird, Bird)
issubclass(SongBird2, Bird)
issubclass(SongBird2, Bird2)
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
#'   - The targetted file is located here: https://caleb-huo.github.io/teaching/data/python/Student_data.csv. 
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
