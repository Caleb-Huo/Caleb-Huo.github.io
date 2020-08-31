#' ---
#' title: "Biostatistical Computing, PHC 6068"
#' subtitle: "Basics about R"
#' author: Zhiguang Huo (Caleb)
#' date: "Mon August 31, 2020"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 
#' 
#' Outline
#' ===
#' 
#' - R data types
#' - R data structure
#' - String manipulation
#' - Loops
#' - Subsetting
#' - File operation
#' 
#' 
#' 
#' Data types
#' ===
#' 
#' - Basic data types in R: 
#'     * logical
#'     * integer
#'     * double
#'     * character
#' 
#' - use typeof() function to check type of a variable.
#'   
#'     
#' Data types (examples)
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
typeof(1.1)
typeof("a")
typeof(TRUE)
typeof(4)
typeof(4L)

#'     
#' 
#' 
#' logical
#' ===
#' 
#' - TRUE/T
#' - FALSE/F
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
if(T){
  print(TRUE)
}

typeof(FALSE)

is.logical(T)

#' 
#' 
#' integer
#' ===
#' 
#' - 1:10 represents integer sequence 1 to 10.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
aseq <- 1:10 ## <- represent assign value
typeof(aseq)

#' - 6L represents 6 is an integer.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
aint = 6L
is.integer(aint)

#' - In R, 6 generally is double instead of integer.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
is.integer(6)

#' 
#' 
#' Assign values
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------------
a = 1.6; print(a) ## assign 1.6 to a 
b <- 1.6; print(b) ## assign 1.6 to b
1.6 -> d; print(d) ## assign 1.6 to d

#' Difference between = and <-
#' 
#' - <- can be only used for value assignment.
#' - = can be used for both value assignment and function argument.
#' 
#' double
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
typeof(1.4142)
is.double(pi)
is.double(0L)
is.double(0L + 1.5)

#' 
#' 
#' character
#' ===
#' 
#' - You can create character using single quotes or double quotes
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
acharacter <- "I like Biostatistical computing"
typeof(acharacter)

bcharacter <- 'You like Biostatistical computing'
is.character(bcharacter)

#' 
#' - When a single quote is part of your string, you need to use double quotes.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
ccharacter <- "He doesn't like Biostatistical computing"
print(ccharacter)

#' 
#' 
#' 1d Vector
#' ===
#' 
#' - Vector is the basic data structure in R. Two types of vectors
#'     * Atomic vector: All elements of an atomic vector must be the same type
#'     * List: elements of a list can be of different type.
#'     
#' - Atomic vectors usually created with c(), short for combine
## ---- eval=FALSE-------------------------------------------------------------------------------------------------------------------------------------
## dbl_var <- c(1, 23.1, 4.2)
## int_var <- c(1L, 11L, 6L)
## log_var <- c(TRUE, FALSE, T, F)
## chr_var <- c("I", "like Biistatistical computing")

#' 
#' 
#' Commonly used vector functions
#' ===
#' 
#' 
#' | Functions 	| Meaning 	|
#' |---	|---	|
#' |length(x)  |  Number of elements in x |
#' |unique(x)  | Unique elements of x   |
#' |sort(x)  |  Sort the elements of x |
#' |rev(x)   | Reverse the order of x   |
#' |names(x)   |  Name the elements of x |
#' |which(x)   | Indices of x that are TRUE   |
#' |which.max(x)   |  Index of the maximum element of x |
#' |which.min(x)   |  Index of the minimum element of x |
#' |append(x)  |  Insert elements into a vector |
#' |match(x)   |  First index of an element in a vector |
#' |union(x, y)  |  Union of x and y |
#' |intersect(x, y)  |  Intersection of x and y |
#' |setdiff(x, y)  |  Elements of x that are not in y |
#' |setequal(x, y)   | Do x and y contain the same elements |                         
#' 
#' 
#' 
#' Example of Vector Functions
#' ===
#' 
#' 
## ---- echo=TRUE, eval=FALSE--------------------------------------------------------------------------------------------------------------------------
## x <- c(1,6,9,2,2,5,10)
## length(x)
## unique(x)
## length(unique(x))
## sort(x)
## rev(x)
## which(x==2)
## which.max(x)
## which.min(x)
## append(x,0)
## match(9,x)
## y <- c(1,2,3)
## union(x,y)
## intersect(x,y)
## setdiff(x,y)
## setequal(x,y)

#' 
#' Statistical Vector Functions
#' ===
#' 
#' | Functions 	| Meaning 	|
#' |---	|---	|
#' |sum(x) |  Sum of x |
#' |prod(x)  | Product of x |
#' |cumsum(x)  |  Cumulative sum of x|
#' |cumprod(x) | Cumulative product of x|
#' |min(x) | Minimum element of x|
#' |max(x) | Maximum element of x|
#' |pmin(x, y) |  Pairwise minimum of x and y |
#' |pmax(x, y) | Pairwise maximum of x and y |
#' |mean(x) |  Mean of x |
#' |median(x) |  Median of x |
#' |var(x) |  Variance of x |
#' |sd(x) |  Standard deviation of x |
#' |cov(x, y) |  Covariance of x and y |
#' |cor(x, y) |  Correlation of x and y |
#' |range(x) |  Range of x |
#' |quantile(x) |  Quantiles of x for given probabilities |
#' |summary(x) |  Numerical summary of x |
#' 
#' 
#' Example of Statistical Vector Functions
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
avec <- c(5,2,9,3)
max(avec)
which.max(avec)
mean(avec)
range(avec)

#' 
## ---- echo=FALSE, eval=FALSE-------------------------------------------------------------------------------------------------------------------------
## x <- c(1,6,9,2,2,5,10)
## y <- c(1,2,3,4,5,6,7)
## sum(x)
## prod(x)
## cumsum(x)
## cumprod(x)
## min(x)
## max(x)
## pmin(x,y)
## pmax(x,y)
## mean(x)
## median(x)
## var(x)
## sd(x)
## cov(x,y)
## cor(x,y)
## range(x)
## quantile(x)
## summary(x)

#' 
#' Coercion
#' ===
#' 
#' - All elements of an atomic vector must be the same type. Otherwise they will be **coerced** to the most flexible type.
#' - Types from least to most flexible are: logical, integer, double and character.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
typeof(c("a", 1))
x <- c(FALSE, FALSE, TRUE)
as.numeric(x)
as.character(x)
typeof(c(1.2,1L))

#' 
#' 
#' How to get help
#' ===
#' 
#' - How to get help of an R function
#'     - ?sum
#'     - help("sum")
#' 
#' - check with the best friend -- google
#' 
#' 
#' Missing value
#' ===
#' 
#' - Missing values are denoted by NA, which is logical vector of length 1.
#' - NA will always be coerced to be the correct type if used inside c()
#' - (optional) You can create NA of a specific type with NA_real_, NA_integer_, NA_character_
## ----------------------------------------------------------------------------------------------------------------------------------------------------
typeof(NA)
typeof(NA_integer_)
typeof(NA_real_)
typeof(NA_character_)

#' 
#' 
#' 
#' list
#' ===
#' 
#' - Lists are different from atomic vectors because their elememts can be of any type, including lists.
#' - Construct lists by using list()      
#' - str() function (short for structure) and gives a compact description of any R data structure.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- list(1:3, "a", c(TRUE, FALSE, TRUE), list(2.3, 5.9))
str(x)
x[[1]]

#' 
#' list
#' ===
#' 
#' - Set names for a list
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- list(p1 = 1:3, p2 = "a", p3 = c(TRUE, FALSE, TRUE), p4 = list(2.3, 5.9))
str(x)
x[[1]]
x$p1

#' 
#' 
#' Data structure
#' ===
#' - Can be organised by their dimensionality (1d, 2d, or nd) and whether they are homogeneous or heterogeneous.
#' 
#' | | Homogeneous 	| Heterogeneous 	|
#' |---  |---	|---	|
#' |1d |  Atomic vector | List |
#' |2d  | Matrix | Data frame |
#' |nd | Array |  |
#' - Note there is no 0-dimensional in R, or scalar types. Individual numbers or strings are acutally vector of length one.
#' 
#' 
#' Attributes
#' ===
#' 
#' - Use to store meta-data.     
#' - can be accessed individually with attr() or attributes().
#' - construct a new object with attributes using structure() function  e.g. 
#'     - structure(1:10, myAttribute="this is a vector") 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
y <- 1:10
attr(y, "my_attribute") <- "This is a vector"
attr(y, "my_attribute")
attributes(y)

#' 
#' Attributes 2
#' ===
#' - Three special attributes have a specific accessor function to get and set values.
#'     - names(): a character vector giving each element a name.
#'     - dim(): used to turn vectors into matrics and arrays.
#'     - class(): used to implement the S3 object system.
#'     
## ----------------------------------------------------------------------------------------------------------------------------------------------------
y <- c(a=1,2:10)
names(y)
names(y)[2] <- 'b'
dim(y)
dim(y) <- c(2,5)
print(y)
class(y)

#' 
#' 
#' factor
#' ===
#' 
#' - A factor is a vector that only contain predefined values.
#' - Factors are used to store categorical data.
#' - Factors are built on top of character vectors using two attributes:
#'     - class(), "factor", which makes them behave differently from regular character vectors. 
#'     - levels(), which defines the set of allowed values.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- factor(c("a", "b", "b", 'a'))
x
class(x)
levels(x)

#' 
#' factors 2
#' ===
#' 
#' factors are very useful when there exist missing class
## ----------------------------------------------------------------------------------------------------------------------------------------------------
sex_char <- c("m", "m", "m")
sex_factor <- factor(sex_char, levels=c("m", "f"))
table(sex_char)
table(sex_factor)

#' 
#' 
#' Matrices
#' ===
#' - Create a matrix with colnames and rownames
## ----------------------------------------------------------------------------------------------------------------------------------------------------
a <- matrix(1:6, ncol=3, nrow=2, dimnames = list(c("row1", "row2"),
                               c("C.1", "C.2", "C.3")))
a
colnames(a)
rownames(a)
ncol(a)
nrow(a)

#' 
#' Matrices
#' ===
#' 
#' - Adding a dim() attribute to an atomic vector
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
c <- 1:6
dim(c) <- c(3,2)
c
dim(c) <- c(2,3)
c

rownames(c) <- c("row1", "row2")
colnames(c) <- c("C.1", "C.2", "C.3")
c

#' 
#' 
#' Data frames
#' ===
#' - A data frame is a very popular way of storing data in R.
#' - A data frame is a list of equal length vector and shares properties of both matrix and list.
#'     - names() and colnames() are the same thing
#'     - length() and ncol() are the same thing
#'     
## ----------------------------------------------------------------------------------------------------------------------------------------------------
df <- data.frame(x=1:3, y=c("a","b","c"),z=0)
str(df)
cat(names(df), "same as", colnames(df))
cat(length(df), "same as", ncol(df))

#' 
#' Data frames 2
#' ===
#' 
#' - data.frame()'s default behaviour turns strings into factors.
#'     - Use stringsAsFactors = FALSE to suppress
#'     - or globally set options(stringsAsFactors=FALSE)
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
df1 <- data.frame(x=1:3, y=c("a","b","c"),z=0)
str(df1)

df2 <- data.frame(x=1:3, y=c("a","b","c"),z=0, stringsAsFactors=FALSE)
str(df2)

#' 
#' 
#' 
#' String manipulation
#' ===
#' - some special characters:
#'     * " " for space
#'     * "\\n" for newline
#'     * "\\t" for tab
## ----------------------------------------------------------------------------------------------------------------------------------------------------
sentenses <- "R is a great statistical software.\n\nWe use R in the Biostatistical computing class!"
sentenses

#' - cat function will recognize these special characters and print to the console:
## ----------------------------------------------------------------------------------------------------------------------------------------------------
cat(sentenses)

#' 
#' 
#' Convert to upper case or lower case
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------------
achar <- "this is a dog."
print(achar)
print(toupper(achar))


print(tolower("WWW.UFL.EDU"))

#' 
#' 
#' Length of a string
#' ===
#' - use nchar to count how many characters in a string instead of length
## ----------------------------------------------------------------------------------------------------------------------------------------------------
achar <- "this is a dog."
nchar(achar)
length(achar)

#' 
#' 
#' vectorizes nchar
#' ===
#' - we can pass a vector of character to nchar
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("a dog", "a cat", "a gator")
nchar(chars)
length(chars)

#' 
#' 
#' Obtaining a substring
#' ===
#' - take a sub-sequence of characters --  use substr(), short for sub string.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- "this is a dog"
substring(chars,1,1)
substring(chars,11,13)

#' 
#' - replace with sub-string.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
substring(chars,11,13) <- "cat"
print(chars)

#' 
#' 
#' strsplit
#' ===
#' - you can split a string by a certain pattern using strsplit() function
## ----------------------------------------------------------------------------------------------------------------------------------------------------
strsplit("this is a dog", split=" ")
strsplit("this is a dog", split="")

#' - Note the return type is a list with only one element. strsplit can be also vectorized.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
strsplit(c("this is a dog", "this is a cat", "this is a gator"), split=" ")

#' 
#' 
#' paste
#' ===
#' - paste multiple strings
## ----------------------------------------------------------------------------------------------------------------------------------------------------
paste('this','is','a','dog', sep=" ")
paste0('this','is','a','dog')

avec <- c('this','is','a','dog')
nchar(avec)
paste(c('this','is','a','dog'), collapse = " ")

#' 
#' 
#' Substituation
#' ===
#' 
#' - use gsub to replace certain pattern within a string.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
achar <- "this is a dog"
gsub(pattern = "dog",replacement="cat",x=achar) ## pattern, replacement, x
gsub("dog","cat",achar) ## pattern, replacement, x

#' 
#' - vectorize
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("this is a dog", "this is a cat", "this is a gator")
gsub("this","that",chars) ## pattern, replacement, x

#' 
#' 
#' Regular expression
#' ===
#' - A regular expression or regex is a structured string to match specific patterns in the text.
#' - grep() function allows us to scan through a vector against regex
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("this is a dog", "this is a cat", "this is a gator")
grep("gator", chars)
grep("this", chars)
grep("gator", chars, value = T)
grep("gator", chars, invert = T)

#' 
#' 
#' Regular expression 2
#' ===
#' - match dog or cat
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("this is a dog", "this is a cat", "this is a gator")
grep("dog|cat", chars)

#' 
#' 
#' Regular expression 3
#' ===
#' - start with
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("abc", "ab", "bc")
grep("b", chars)
grep("^b", chars)

#' 
#' - end with
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("abc", "ab", "bc")
grep("ab", chars)
grep("ab$", chars)

#' 
#' 
#' Metacharacters
#' ===
#' 
#' - **Metacharacters** are special characters with a special meaning.
#' - A regular expression often consists metacharacters
#' - Square braces are used to match anything in the braces
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("this is a dog", "this is a cat", "this is a gator")
grep("[bced]", chars)

#' - dash inside square braces is used to indicate a range
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("this is a dog", "this is a cat", "this is a gator")
grep("[b-d]", chars)
grep("[0-9]", chars)

#' 
#' 
#' Metacharacters 2
#' ===
#' - "[:alnum]" matches any alphanumeric character, same as "[a-zA-Z0-9]"
#' - "[:punct:]" matches to any punctuation mark
#' - "[:space:]" matches to any white space character (tab and line break).
#' - A caret inside braces matches anything except the followng words.
#'     - "[^0-9]" matches anything but a number between 0 and 9.
#'     - "[^aeiou]" matches anything but a lower case vowel.
#' - A period "." matches to any character.
#' 
#' Substituation using metacharacters
#' ===
#' 
#' - use gsub to replace certain pattern within a string.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
chars <- c("this is a dog", "this is a cat", "this is a   gator")
gsub(pattern = "[aeiou]",replacement="Z",x=chars) ## pattern is vowel, replacement is Z
gsub("[0-9]","#","a1b2") ## pattern, replacement, x

#' 
#' 
#' Loop
#' ===
#' - for loop
## ----------------------------------------------------------------------------------------------------------------------------------------------------
for(i in 1:10){
  cat(i," ")
}

#' 
#' - while loop
## ----------------------------------------------------------------------------------------------------------------------------------------------------
i <- 1
while(i <= 10){
  cat(i," ")
  i <- i + 1
}

#' 
#' 
#' Loop 2
#' ===
#' - next: directly start next round of loop
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
for(i in 1:10){
  if(i%%2==0){
    next
  }
  cat(i," ")
}

#' 
#' - equivalently: 
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
for(i in seq_len(10)){
  if(i%%2==0){
    next
  }
  cat(i," ")
}

#' 
#' - break: break the loop
## ----------------------------------------------------------------------------------------------------------------------------------------------------
for(i in 1:10){
  if(i%%2==0){
    break
  }
  cat(i," ")
}

#' 
#' Loop 3
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- list(1:3, "a", c(TRUE, FALSE, TRUE), list(2.3, 5.9))
for(i in 1:length(x)){ ## 1:length(x) is not recommended
  ax <- x[[i]]
  print(ax)
}

#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
for(i in seq_along(x)){ ## seq_along(x) is the same as 1:length(x) 
  ax <- x[[i]]
  print(ax)
}

#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
for(ax in x){ ## direct sub-element
  print(ax)
}

#' 
#' Subsetting -- Atomic vectors 1
#' ===
#' 
#' example: x <- c(2.1, 4.2, 3.3, 5.4). How can we obtain a subset of this vector?
#' 
#' - Positive integers: return elements at the specified position.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
### subseting
## atomic vectors
x <- c(2.1, 4.2, 3.3, 5.4)

# Positive integer
x[c(3,1)]
x[order(x)] ## equivalent to sort(x)
x[c(1,1,1)]

#' 
#' Subsetting -- Atomic vectors 1
#' ===
#' example: x <- c(2.1, 4.2, 3.3, 5.4). How can we obtain a subset of this vector?
#' 
#' - Negative integers: omit elements at the specified positions.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
# negative integer
x[-c(1, 3)]

x[-grep(4.2, x)]


#' 
#' Subsetting -- Atomic vectors 3
#' ===
#' example: x <- c(2.1, 4.2, 3.3, 5.4). How can we obtain a subset of this vector?
#' 
#' - Logical vectors: select elements where the corresponding logical value is TRUE.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x[c(TRUE, TRUE, FALSE, FALSE)]
x > 3
x[x > 3]
x[c(TRUE, TRUE, NA, FALSE)]

#' 
#' Subsetting -- Atomic vectors 4
#' ===
#' example: x <- c(2.1, 4.2, 3.3, 5.4). How can we obtain a subset of this vector?
#' 
#' - Nothing: return the original vector.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x[]

#' 
#' - Zero: return a zero length vector.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x[0]

#' 
#' Subsetting -- Atomic vectors 5
#' ===
#' example: x <- c(2.1, 4.2, 3.3, 5.4). How can we obtain a subset of this vector?
#' 
#' - Character vectors: to return elements with matching names.
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- c(a=2.1, b=4.2, c=3.3, d=5.4)

x["a"]
x[letters[2:3]]

#' 
#' 
#' 
#' Subsetting -- Matrices and arrays
#' ===
#' 
#' - 1d index for each dimension, separated by comma
## ----------------------------------------------------------------------------------------------------------------------------------------------------
a <- matrix(1:9, nrow=3)
colnames(a) <- c("A","B","C")
a
a[1:2,]
a[c(T,F,T), c("B","A")]
a[,-2]

#' 
#' 
#' Subsetting -- Data frame
#' ===
#' - Data frames possess the characteristics of both lists and matrices.
## ----------------------------------------------------------------------------------------------------------------------------------------------------
options(stringsAsFactors = FALSE)
df <- data.frame(x=1:2, y=2:1, z=letters[1:2])
df[df$x==2,]
df[c("x","z")] # like a list
df[,c("x","z")] # like a matrix

#' 
#' Subsetting -- simplifying vs preserving
#' ===
#' - two types of subsetting: simplifying and preserving subsetting.
#'     - Simplifying subsets returns the simplest possible data structure that can represent the output.
#'     - Preserving subsetting keeps the structure of the output the same as the input.
#' 
#' | Functions 	| simplifying 	| preserving | 
#' |---	|---	|---  |
#' | List | x[[1]] | x[1] |
#' | Vector | x[[1]] | x[1] |
#' | Factor | x[1:2, drop=T] | x[1:2] |
#' | Data frame | x[,1] or x[[1]] | x[, 1, drop=F] or x[1] |         
#'          
#' 
#' Subsetting -- simplifying vs preserving
#' ===
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
x <- list(a=111, b=222)


x[[1]] ## simplified, result is numeric
typeof(x[[1]])

x[1] ## preserved, result is a list
typeof(x[1])


#'          
#' Matching
#' ===
## ----------------------------------------------------------------------------------------------------------------------------------------------------
grades <- c(1,2,2,3,1)
info <- data.frame(grade=3:1, desc=c("Excellent", "Good", "Poor"), fail=c(F,F,T))
id <- match(grades, info$grade)
id
info[id,]

#' 
#' 
#' File I/O
#' ===
#' 
#' Read in txt/csv files
#' 
#' - read in locally
#' 
## ---- eval = F---------------------------------------------------------------------------------------------------------------------------------------
## burnData <- read.csv("burn.csv", row.names = 1)

#' 
#' - read directly from URL
#' 
## ----------------------------------------------------------------------------------------------------------------------------------------------------
burnData <- read.csv("https://caleb-huo.github.io/teaching/data/Burn/burn.csv", row.names = 1)

#' 
#' - read only the first two lines
#' 
## ---- eval = F---------------------------------------------------------------------------------------------------------------------------------------
## setwd("~/Desktop") ## set your working directory
## burnData <- read.csv("burn.csv", row.names = 1, nrows = 2)

#' 
#' - skip the first three lines
#' 
## ---- eval = F---------------------------------------------------------------------------------------------------------------------------------------
## burnData <- read.csv("burn.csv", row.names = 1, skip = 3)

#' 
#' Delimiter
#' ===
#' 
#' - comma: 
#' ```
#' read.csv()
#' ```
#' 
#' - tab: 
#' ```
#' read.table()
#' ```
#' 
#' - specify your own: 
#' ```
#' read.delim(, delim=";")
#' ```
#' 
#' Also pay attention to the arguments such as header, row.names
#' 
#' Save txt/csv files
#' ===
#' 
#' ```
#' write.csv(burnData, file = "myBurnData.csv")
#' write.table(burnData, file = "myBurnData.txt")
#' write.table(burnData, file = "myBurnData.txt", append = TRUE)
#' ```
#' 
#' Read in line by line
#' ===
#' 
#' ```
#' fileNameFull <- 'https://caleb-huo.github.io/teaching/data/Burn/burn.csv'
#' con  <- file(fileNameFull, open = "r")
#' 
#' while (length(oneLine <- readLines(con, n = 1, warn = FALSE)) > 0) {
#' 	aline = strsplit(oneLine, ",")[[1]]
#' 	print(aline)
#' } 
#' close(con) ## remember to close files
#' ```
#' 
#' Save R objects
#' ===
#' If you take a long time to obtain your result.
#' How to save your result so in the future, you won't bother re-run them again?
#' 
#' - Create a list for these variables
#' - Save the list by save()
#' 
#' ```
#' a <- 1:4
#' b <- 2:5
#' ans <- a * b
#' result <- list(a=a, b=b, ans=ans)
#' save(result,file="myResult.rdata")
#' ```
#' 
#' 
#' References
#' ===
#' 
#' - [http://adv-r.had.co.nz](http://adv-r.had.co.nz)
#' - [http://www.stat.cmu.edu/~ryantibs/statcomp/](http://www.stat.cmu.edu/~ryantibs/statcomp/)
#' 
