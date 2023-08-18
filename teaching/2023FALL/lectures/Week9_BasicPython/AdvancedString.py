#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday Oct 19th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Advanced string manupulation"
#' ---
#' 

#' 
#' 
#' 
#' Outlines
#' ===
#' 
#' - Review of basic string operation
#' - Regular expressions
#' - Goal: Using a single regex to identify complex patterns like
#'     - doi:10.1038/nphys1170
#'     - doi:10.1002/0470841559.ch1
#'     - DOI:10.1093/bib/bbab224
#' 
#' 
#' 
#' 
#' Review of basic string operation
#' ===
#' 
title = "I love introduction to Biostatistical computing!"
title.find("love")
title.find("XX")
title.index("love")

title.replace("love", "like")
#' 
#' 
#' Regular expression (regex)
#' ===
#' 
#' - Regular expressions
#'     - called REs, or regexes, or regex patterns
#'     - available in python through re module
#'     - specify the rules for the set of possible strings that you want to match;
#' 
import re ## import the python regular expression module
#' 
#' - re module methods
#'     - search/match
#'     - split
#'     - findall
#'     - sub
#'     
#' re.search / re.match
#' ===
#' 
#' - the match method looks for some pattern and returns a Boolean
#' 
text = "Have a nice day!"
re.match("Have", text)
#' 
#' - the re.match result is a Boolean
#' 
amatch = re.match("Have", text)
if amatch:
    print("Wonderful!")
else:
    print("Oops")
#' 
#' ---
#' 
#' - the span method will return the match index
#' 
amatch = re.match("Have", text)
print(amatch.span())
#' 
#' - the group method will return the content
#' 
amatch = re.match("Have", text)
print(amatch.group())
#' 
#' 
#' split
#' ===
#' 
#' - the split method will use a pattern for creating a list of substrings
#' 
text = "Alex works diligently. Alex gets good grades. Our student Alex is successful"

re.split("Alex", text) ## regular expression

text.split("Alex") ## traditional method


#' 
#' - So far, their behaviours are the same.
#' - re.split can handle regex, while the regular split method cannot.
#' 
#' 
#' findall
#' ===
#' 
#' - The findall methods will look for a pattern and pull out all occurences.
#' 
text = "Alex works diligently. Alex gets good grades. Our student Alex is successful"

re.findall("Alex", text)
re.findall("student", text)
re.findall(" s", text)
re.findall("s ", text)
#' 
#' 
#' Complex patterns
#' ===
#' 
#' - Anchors
#'     - start: ^
#'     - end: $
#' 
grades = "ACAAAABCBCBAA"
re.findall("A", grades)
re.findall("^A", grades)
re.findall("A$", grades)
#' 
#' - Will use findall to demonstrate.
#' - The pattern rule works for other RE methods.
#' 
#' 
#' Complex patterns
#' ===
#' 
#' - `[]` is called set operator. A range of character.
#' 
#' - `[AB]`: A or B
#' 
grades = "ACAAAABCBCBAA"
re.findall("[AB]", grades) ## A or B
#' 
#' - any upper case letter
#' 
content = "I love biostatistical computing"
re.findall("[A-Z]", content) 
re.findall("[A-Z]", content.title()) 
#' 
#' Complex patterns
#' ===
#' 
#' - any lower case letter
#' 
content = "I love biostatistical computing"
re.findall("[a-z]", content.title()) 
#' 
#' - any digits
#' 
content = "A1B2C3D4"
re.findall("[0-9]", content) 
#' 
#' 
#' Complex patterns
#' ===
#' 
#' - \\w: any letter or word
#' - \\d: any digits
#' - \\s: Matches Unicode whitespace characters, which includes white space, \\t, \\n, etc.
#' 
content = "A1B2C3 D4\n E5\t"
re.findall("\w", content) 
re.findall("\d", content) 
re.findall("\s", content) 
#' 
#' 
#' 
#' Complex patterns 
#' ===
#' 
#' - |: or pattern
#' 
grades = "ACAAAABCBCBAA"
re.findall("A|B", grades) ## A or B
#' 
#' 
grades = "ACAAAABCBCBAA"
re.findall("[A][B-C]", grades)
### [A][B-C] pattern denoted two sets of characters which must have been matched back to back.
#' 
#' - Equivalent as AB|AC
#' 
grades = "ACAAAABCBCBAA"
re.findall("AB|AC", grades) ## AB or AC
#' 
#' 
#' Complex patterns: do not match [^]
#' ===
#' 
grades = "ACAAAABCBCBAA"
re.findall("[^A]", grades) ## in the [], not A
#' 
#' - The starting letter is not A 
#' 
grades = "ACAAAABCBCBAA"
re.findall("^[^A]", grades) ## ^ in the [], not A
#' 
#' - The starting letter is not B 
#' 
grades = "ACAAAABCBCBAA"
re.findall("^[^B]", grades) 
#' 
#' Match telephone numbers, in class practice
#' ===
#' 
cell_numbers = "123 456-7890, 321-654-7890, (213)456-7980, (123)-456-0987, 1254367890, 123-4587690"
#' 
#' - We only want to select **one** of the following patterns
#'     1. 321-654-7890
#'     2. (213)456-7980
#'     
#' - Hint: 
#'     - We can do match one patterns at a time.
#'     - Since \( has special meaning in regex, need to use \\ \( to indicate the literal \(
#' 
#' 
#' ---
#' 
re.findall("\(\d{3}\)\d{3}-\d{4}", cell_numbers)
re.findall("\d{3}-\d{3}-\d{4}", cell_numbers)
#' 
#' 
#' 
#' Quantifiers
#' ===
#' 
#' - Quantifiers are the number of times you want a pattern to be matched in order to match.
#' 
#' 
grades = "ACAAAABCBCBAA"
re.findall("A{2,10}", grades) ## 2: minimum number of A; 10: maximum number of A
#' 
#' - how many times has this student been on a back to back A's streak?
#' 
#' - two A's back to back
#' 
grades = "ACAAAABCBCBAA"
re.findall("A{1,1}A{1,1}", grades)
#' 
#' 
#' ---
#' 
#' 
#' - Regex quantifier syntax does not allow you to deviate from {m,n}.
#' - In particular, in you have an extra whitespece, you'll get an empty result
#' 
grades = "ACAAAABCBCBAA"
re.findall("A{2, 2}", grades)
#' 
#' - if have one number in the braces, it's considered to be both m and n.
#' 
grades = "ACAAAABCBCBAA"
re.findall("A{2}", grades)
#' 
#' -  find a decreasing trend in a student's grades
#' 
re.findall("A{1,10}B{1,10}C{1,10}", grades)
#' 
#' 
#' Make the maximum arbitrarily large
#' ===
#' 
#' - \* to match 0 or more times of the preceding RE,
#' - \+ to match one or more times of the preceding RE.
#' - \? to match 0 or 1 repetitions of the preceding RE
#' 
grades = "ACAAAABCBCBAA"
re.findall("A+B+C+", grades)
re.findall("A?B?C?", grades)
re.findall("A*B+", grades)
#' 
#' Remove extra whitespaes, in class exercise
#' ===
#' 
#' - How to remove the extra \\n and whitespace?
#'     - no whitespace (or \\n) at the head or the end
#'     - single space between two words
#' 
text = " Alex    works diligently.  \n\n  Alex   gets   good grades. Our student   Alex is successful  \n"
#' 
#' 
#' ---
#' 
#' - solution
#' 
re.sub("\s+", " ", text.strip())
#' 
#' 
#' 
#' Examples
#' ===
#' 
#' - https://en.wikipedia.org/wiki/Family_Educational_Rights_and_Privacy_Act
#' - https://caleb-huo.github.io/teaching/data/Python/ferpa.txt
#' 
with open("ferpa.txt", "r") as file:
    wiki = file.read()

print(wiki)
#' 
#' ---
#' 
#' - to identify section header:
#' 
re.findall("[a-zA-Z]{1,100}\[edit\]", wiki)
#' 
#' - Since \[ has special meaning, need to use \\ \[ to indicate the literal \[
#' 
#' - \\w to match any letter or digit
re.findall("[\w]{1,100}\[edit\]", wiki)
#' 
#' - alternatively
#' 
re.findall("[\w]+\[edit\]", wiki)
#' 
#' ---
#' 
#' - to have the entire title with edit:
#' 
re.findall("[\w ]*\[edit\]", wiki)
#' 
#' - To print the real title line by line:
#' 
for title in re.findall("[\w ]*\[edit\]", wiki):
    print(re.split("[\[]", title)[0])
#' 
#' - Is there a simple way to do this?
#' 
#' Group
#' ===
#' 
#' - in regex, we could specify groups by ()
#' -  return multiple groups in tuple.
#' 
re.findall("([\w ]*)(\[edit\])", wiki)
#' 
for title in re.findall("([\w ]*)(\[edit\])", wiki):
    print(title[0])
#' 
#' 
#' iterator
#' ===
#' 
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))
#' 
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

for i in myit:
  print(i)
#' 
#' 
#' create a iterator for RE
#' ===
#' 
for item in re.finditer("([\w ]*)(\[edit\])", wiki):
    print(item.groups())
#' 
#' 
item
#' 
#' --- 
#' 
item.span()
item.group()
item.group(0)
item.group(1)
item.group(2)
#' 
#' Add title to groups
#' ===
#' 
#' - ?P<>: group labels
#' 
for item in re.finditer("(?P<title>[\w ]*)(?P<edit_link>\[edit\])", wiki):
    print(item.groupdict()['title'])
#' 
#' - With labels, the resulting item can be converted to a dictionary
#' 
item.groupdict()
#' 
#' 
#' Look-ahead and look-behind
#' ===
#' 
#' - lookahead ?=: 
#'     - (a)(?=b), will match the pattern ab, but only print a (ahead).
#'     
text = "Amy works diligently. Alex gets good grades. Our student Aaron is successful"
re.findall("(\w+ )(?=works)", text)
#' 
#' - lookbehind ?<=:
#'     - (?<=a)(b), will match the pattern ab, but only print b (behind).
#' 
m = re.search('(?<=abc)def', 'abcdef')
m.group()
#' 
m = re.search('(?=abc)def', 'abcdef')
m
#' 
#' 
#' Example, Twitter data
#' ===
#' 
#' - https://caleb-huo.github.io/teaching/data/Python/nytimeshealth.txt
#' 
with open("nytimeshealth.txt", "r") as file:
    health=file.read()
print(health[:600])
#' 
#' - how to find all users? (i.e., @)
#' - how to find all dates (where a @user appears)
#'     - only print the dates (e.g., 'Fri Dec 26')
#' 
#' ---
#' 
re.findall("@\w+", health)
#' 
#' ---
#' 
# multiple experiments
# re.findall("(\w+ \w+ \d+)(.*@\w+)", health)
# re.findall("(\w+ \w+ \d+)(?=.*@\w+)", health)
# re.findall("(\w{3} \w{3} \d+)(?=.*@\w+)", health)
re.findall("(?<=\|)(\w{3} \w{3} \d+)(?=.*@\w+)", health)
#' 
#' 
#' Nested Repetition Quantifiers
#' ===
#' 
#' - ?: will treat as a group internally, won't print as a group.
#'     - If you want to repeat a quantifier, you have to use ?:
#' 
text_doi = "doi:10.1038/nphys1170, doi:10.1002/0470841559.ch1, DOI:10.1093/bib/bbab224"

re.findall("DOI|doi", text_doi)
re.findall("[Dd][Oo][Ii]:\d+", text_doi)
re.findall("(?:DOI|doi):\d+", text_doi)
#' 
#' how to match the entire doi pattern?
#' 
#' ---
#' 
#' 
text_doi = "doi:10.1038/nphys1170, doi:10.1002/0470841559.ch1, DOI:10.1093/bib/bbab224"

re.findall("(?:DOI|doi):\d+", text_doi)
re.findall("(?:DOI|doi):\d+\.\d+/(?:\w+\.?)+", text_doi)
re.findall("(?:DOI|doi):\d+\.\d+(?:/(?:\w+\.?)+)+", text_doi)
#' 
#' Reference
#' ===
#' 
#' - https://docs.python.org/3/index.html
#' - https://www.coursera.org/learn/python-data-analysis?specialization=data-science-python
