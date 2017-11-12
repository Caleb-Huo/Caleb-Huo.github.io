
x <- seq(0, 10, 2); y <- x + rnorm(length(x))
dat <- rbind(y, x)
attr(dat, "class") <- "foo"
# class(dat) <- "foo" #### Alternatively, use class()
# dat <- structure(rbind(y, x), class = "foo")
dat

attr(dat, "c.names") <- paste("C", 1:length(x), sep='') # Add new attributes
dat

head(methods(print))

print.foo <- function(x, ...) {
    y <- data.frame(matrix(x, nrow(x)))
    colnames(y) <- attr(x, "c.names")
    
    cat("Print 'foo' \n")
    print(y)
}
print(dat)

### Traditional programming, BMI
weight <- 62
size <- 1.70
(BMI <- weight/size**2)

### Traditional programming, my BMI
weightMe <- 62
sizeMe <- 1.70
(BMIMe <- weightMe/sizeMe**2)

### Traditional programming, his BMI
weightHim <- 85
sizeHim <- 1.84
(BMIHim <- weightMe/sizeHim**2)

### Definition of an object BMI
setClass("BMI", slots = list(weight = "numeric", size = "numeric"))
setMethod("show", "BMI",
          function(object) {
            cat("BMI =", object@weight/object@size**2, " \n")
          }
)

(myBMI <- new("BMI", weight = 62, size = 1.70))

(hisBMI <- new("BMI", weight = 85, size = 1.84))

(weight <- "hello")

new("BMI", weight="hello", size=1.70)

(sizeMe <- -1.70)

### Object programming, control
setValidity("BMI",
            function(object) {
              if (object@size < 0) {
                return("Negative size not accepted. ")
              } else TRUE
            }
)
new("BMI", weight = 62, size = -1.7)

### Definition of the heir
setClass("BMIplus", slots = list(sex = "character"), contains = "BMI")
she <- new("BMIplus", size = 1.65, weight = 52, sex = "Female")
she

#        ------------------------------------------
#        |                BMI                     |         --------------------
#        |----------------------------------------|         |  BMIplus         |
#        |    Slots     |                         |         |------------------|
#        |---------------                         |         | Slots  |         |
#        |      weight: numeric                   |  <---   |---------         |
#        |      size:   numeric                   |         |   sex: character |
#        |----------------------------------------|         |------------------|
#        |   Methods    |                         |         | Methods|         |
#        |---------------                         |         |---------         |
#        |      NULL <- show(object: BMI)         |         --------------------
#        ------------------------------------------

#      --------------------------         ----------------------
#      |     bmiTraj            |         |      ptGroup       |
#      --------------------------         ----------------------
#      | Slots |  week          |         | Slots |   nbGroups |
#      ---------  traj          |         ---------   group    |
#      --------------------------         ----------------------
#      | Methods | countMissing |         |         | Methods  |
#      ----------- print        |         ----------------------
#      |           imputation   |             ^
#      -----------------------                |
#                          /\                 |
#                          ||                 |
#                      ---------------------------
#                      |      bmiGroups          |
#                      ---------------------------
#                      | Slots |                 |
#                      --------| groupList       |
#                      ---------------------------
#                      | Methods |               |
#                      -----------  print        |
#                      ---------------------------

setClass(
  Class = "bmiTraj",
  slots = list(
    week = "numeric",
    traj = "matrix"
  )
)

new(Class = "bmiTraj")
# new(Class = "bmiTraj", week = c(1, 3, 4))
# new(Class = "bmiTraj", week = c(1, 3), traj = matrix(1:4, 2, 2))

# Three doctors take part to the study. Bob, Vic and Ana
bmiBob <- new(Class = "bmiTraj")
bmiVic <- new(
  Class = "bmiTraj",
  week = c(1, 3, 4, 5),
  traj = rbind(c(15, 15.1, 15.2, 15.2),
               c(16, 15.9, 16, 16.4),
               c(15.2, NA, 15.3, 15.3),
               c(15.7, 15.6, 15.8, 16))
)
bmiAna <- new(
  Class = "bmiTraj",
  week = c(1:10, 6:16*2),
  traj = rbind(matrix(seq(16, 19, length = 21), 50, 21, byrow = TRUE),
               matrix(seq(15.8, 18, length = 21), 30, 21, byrow = TRUE)) + rnorm(21*80, 0, 0.2)
)

bmiVic@week
bmiVic@week <- c(1, 2, 4, 5)
bmiVic

setClass(
  Class = "bmiTrajNew",
  slots = list(
    week = "numeric",
    traj = "matrix"
  ),
  prototype = prototype(week = 1, traj = matrix(0))
)

removeClass("bmiTrajNew")
new("bmiTrajNew")

slotNames("bmiTraj")

getSlots("bmiTraj")

getClass("bmiTraj")

size <- rnorm(10, 1.70, 0.1)
weight <- rnorm(10, 70, 5)
group <- as.factor(rep(c("A", "B"), 5))
options(repr.plot.width=7, repr.plot.height=3)
par(mfrow = 1:2)
plot(size ~ weight)
plot(size ~ group)

setMethod(f = "plot", signature = "bmiTraj",
          definition = function(x, y, ...) {
            matplot(x@week, t(x@traj), xaxt="n", type="l", ylab="", xlab="", pch=1)
            axis(1, at=x@week)
          }
)
par(mfrow = 1:2); plot(bmiVic); plot(bmiAna)

# Note: during the redefinition of a function, R imposes to use the same arguments as the function in question. 
# To know the arguments of the 'plot', one can use 'args'
args(plot)
args(print)

setMethod("print", "bmiTraj",
          function(x, ...) {

            cat("*** Class bmiTraj, method Print *** \n")
            cat("* Week = "); print(x@week)
            cat("* Traj = \n"); print(x@traj)
            cat("******* End Print (bmiTraj) ******* \n")
          }
)
print(bmiVic)

setMethod("show", "bmiTraj",
          function(object) {
            cat("*** Class bmiTraj, method show *** \n")
            cat("* Week = "); print(object@week)
            nrowShow <- min(10, nrow(object@traj))
            ncolShow <- min(10, ncol(object@traj))
            cat("* Traj (limited to a matrix 10x10) = \n")
            print(formatC(object@traj[1:nrowShow, 1:ncolShow]), quote = FALSE)
            cat("******* End Show (bmiTraj) ******* \n")
          }
)
bmiAna

new("bmiTraj")

setMethod("show", "bmiTraj",
          function(object) {

            cat("*** Class bmiTraj, method show *** \n")
            cat("* Week = "); print(object@week)
            nrowShow <- min(10, nrow(object@traj))
            ncolShow <- min(10, ncol(object@traj))
            cat("* Traj (limited to a matrix 10x10) = \n")
            if(length(object@traj) != 0)
              print(formatC(object@traj[1:nrowShow, 1:ncolShow]), quote = FALSE)
            cat("******* End Show (bmiTraj) ******* \n")
          }
)
new("bmiTraj")

setGeneric(name = "countMissing", def = function(object) standardGeneric("countMissing"))
setMethod(
  f = "countMissing",
  signature = "bmiTraj",
  definition = function(object) return(sum(is.na(object@traj)))
)
countMissing(bmiVic)

showMethods(classes = "bmiTraj")

# getMethod enables to see the definition (the contents of the body function) of a method for a given object
getMethod(f = "plot", signature = "bmiTraj")

getMethod(f = "plot", signature = "bmiTraj")

existsMethod(f = "plot", signature = "bmiTraj")

existsMethod(f = "plot", signature = "bmiTrej")

setClass(
  Class = "bmiTraj",
  slots = list(week = "numeric", traj = "matrix"),
  validity = function(object) {

    cat("--- bmiTraj: inspector --- \n")
    if(length(object@week) != ncol(object@traj))
      stop("[bmiTraj: validation] the number of weeks does not correspond to the number of columns of the matrix")
    return(TRUE)
  }
)

new(Class = "bmiTraj", week = 1:2, traj = matrix(1:2, ncol = 2))

new(Class = "bmiTraj", week = 1:3, traj = matrix(1:2, ncol = 2))

# The inspector will not be called after the creation of the object 
bmiPoppy <- new(Class = "bmiTraj", week = 1, traj = matrix(1))
(bmiPoppy@week <- 1:3)

setMethod(f = "initialize", signature = "bmiTraj",
          definition = function(.Object, week, traj) {
            cat("--- bmiTraj: initializator --- \n")
            rownames(traj) <- paste("I", 1:nrow(traj), sep='')
            .Object@traj <- traj
            .Object@week <- week
            return(.Object)
          }
)
new(Class = "bmiTraj", week = c(1, 2, 4, 5), traj = matrix(1:8, nrow = 2))

new(Class = "bmiTraj", week = c(1, 2, 4), traj = matrix(1:8, nrow = 2))

# To use an initializator and an inspector in the same object, it is thus necessary to call "manually" the inspector
setMethod(f = "initialize", signature = "bmiTraj",
          definition = function(.Object, week, traj) {
            cat("--- bmiTraj: initializator --- \n")
            if(!missing(traj)) {
              colnames(traj) <- paste("T", week, sep='')
              rownames(traj) <- paste("I", 1:nrow(traj), sep='')
              .Object@traj <- traj
              .Object@week <- week
              validObject(.Object)      # call of the inspector
            }
            return(.Object)
          }
)
new(Class = "bmiTraj", week = c(1, 2, 48), traj = matrix(1:8, nrow = 2))

tr <- bmiTraj <- function(week, traj) {

  cat("----- bmiTraj: constructor ----- \n")
  new(Class = "bmiTraj", week = week, traj = traj)
}
bmiTraj(week = c(1, 2, 4), traj = matrix(1:6, ncol=3))

tr <- bmiTraj <- function(week, traj) {

  if(missing(week)) week <- 1:ncol(traj)
  cat("----- bmiTraj: constructor ----- \n")
  new(Class = "bmiTraj", week = week, traj = traj)
}
bmiTraj(traj = matrix(1:8, ncol=4))

regularBmiTraj <- function(nbWeek, BMIinit) {

  traj <- outer(BMIinit, 1:nbWeek, function(init, week) return(init+0.1*week))
  week <- 1:nbWeek
  return(new(Class = "bmiTraj", week = week, traj = traj))
}
regularBmiTraj(nbWeek = 3, BMIinit = 14:16)

setGeneric("getWeek", function(object) standardGeneric("getWeek"))
setMethod("getWeek", "bmiTraj", function(object) return(object@week))
getWeek(bmiVic)

setGeneric("getTraj", function(object) standardGeneric("getTraj"))
setMethod("getTraj", "bmiTraj", function(object) return(object@traj))
getTraj(bmiVic)

setGeneric("setWeek<-", function(object, value) standardGeneric("setWeek<-"))
setReplaceMethod(f = "setWeek", signature = "bmiTraj",
  definition = function(object, value) {
    object@week <- value
    return(object)
  }
)
getWeek(bmiVic)

setWeek(bmiVic) <- 1:3
getWeek(bmiVic)

setMethod(
  f = "[",
  signature = "bmiTraj",
  definition = function(x, i, j, drop) {

    if(i == "week") return(x@week)
    if(i == "traj") return(x@traj)
  }
)
bmiVic["week"]
bmiVic["traj"]

setReplaceMethod(
  f = "[",
  signature = "bmiTraj",
  definition = function(x, i, j, value) {

    if(i == "week") x@week <- value
    if(i == "traj") x@traj <- value
    validObject(x)
    return(x)
  }
)
bmiVic["week"] <- 2:5
bmiVic["week"]

setClass(
  Class = "ptGroup",
  slots = list(
    nbGroups = "numeric",
    group = "factor"
  )
)
setGeneric("getNbGroups", function(object) standardGeneric("getNbGroups"))
setMethod("getNbGroups", "ptGroup", function(object) return(object@nbGroups))
setGeneric("getGroup", function(object) standardGeneric("getGroup"))
setMethod("getGroup", "ptGroup", function(object) return(object@group))    
groupVic <- new(Class = "ptGroup", nbGroups = 2, group = factor(c("A", "B", "A", "B")))
groupAna <- new(Class = "ptGroup", nbGroups = 2, group = factor(rep(c("A", "B"), c(50, 30))))

setGeneric("test", function(x, y, ...) standardGeneric("test"))
setMethod("test", "numeric", function(x, y, ...) cat("x is numeric =", x, "\n"))

test(3.17)

test("E")

setMethod("test", "character", function(x, y, ...) cat("x is character =", x, "\n"))
test("E")

# More complicated, we wish that test shows a different behavior if one combines a numeric and a character.
setMethod(
  f = "test",
  signature = c(x = "numeric", y = "character"),
  definition = function(x, y, ...) {
    cat("more complicated: ")
    cat("x is numeric =", x, "AND y is a character =", y, "\n")
  }
)
test(3.2, "E")

test(3.2)
test("E")

options(repr.plot.width=7, repr.plot.height=3)
par(mfrow = c(1, 2))
plot(bmiVic); plot(bmiAna)

setMethod(
  f = "plot",
  signature = c(x = "bmiTraj", y = "ptGroup"),
  definition = function(x, y, ...) {
    matplot(x@week, t(x@traj[y@group == levels(y@group)[1], ]), ylim = range(x@traj, na.rm = TRUE),
            xaxt = "n", type = "l", ylab = "", xlab = "", col = 2)
    for(i in 2:y@nbGroups) {
      matlines(x@week, t(x@traj[y@group == levels(y@group)[i], ]), xaxt = "n", type = "l", col = i+1)
    }
    axis(1, at = x@week)
  }
)
par(mfrow = c(1, 2))
plot(bmiVic, groupVic)
plot(bmiAna, groupAna)

showMethods(test)

test(1, TRUE)

setMethod(
  f = "test",
  signature = c(x = "numeric", y = "missing"),
  definition = function(x, y, ...) cat("x is numeric =", x, "and y is 'missing' \n")
)
test(3.17)
test(3.17, "E")
test(3.17, TRUE)

#                         -----------------------
#                         |        ANY          |
#                         -----------------------
#                          ^                   ^
#                          |                   |
#                 -----------------     -----------------
#                 |  Father A     |     |  Father B     |
#                 -----------------     -----------------
#                  ^            ^             ^
#            -----------  -----------      -----------
#            |  Son A1 |  |  Son A2 |      |  Son B1 |
#            -----------  -----------      -----------
#                  ^
#            --------------
#            | Gd son A1a |
#            --------------

setClass(
  Class = "bmiGroups",
  slots = list(groupList = "list"),
  contains = "bmiTraj"
)
bgLisa <- new("bmiGroups")

bgLisa

unclass(bgLisa)

groupVic2 <- new("ptGroup", nbGroups = 3, group = factor(c("A", "C", "C", "B")))
bgVic <- new(
  Class = "bmiGroups",
  week = c(1, 3, 4, 5),
  traj = bgVic@traj,
  groupList = list(groupVic, groupVic2)
)

getMethod("initialize", "bmiGroups")

existsMethod("initialize", "bmiGroups")

hasMethod("initialize", "bmiGroups")

selectMethod("initialize", "bmiGroups")

setMethod("initialize", "bmiGroups",
          function(.Object, week, traj, groupList) {
            cat("---- groupList: initializator ---- \n")
            if(!missing(traj)) {
              .Object@week <- week
              .Object@traj <- traj
              .Object@groupList <- groupList
            }
            return(.Object)
          }
)
bgVic <- new(
  Class = "bmiGroups",
  week = c(1, 3, 4, 5),
  traj = bmiVic@traj,
  groupList = list(groupVic, groupVic2)
)

print(bgVic)

setMethod(
  f = "print",
  signature = "bmiGroups",
  definition = function(x, ...) {

    callNextMethod()    #### callNextMethod()
    cat("the object also contains", length(x@groupList), "groups. \n")
    cat("**** Fine of print (bmiGroups) **** \n")
    return(invisible())
  }
)
print(bgVic)

print(as(bgVic, "bmiGroups"))

# That will be useful to us in the definition of 'show for bmiGroups, no need to use callNextMethod
setMethod(
  f = "show",
  signature = "bmiGroups",
  definition = function(object) {

    show(as(object, "bmiTraj"))  ## Instead of callNextMethod, use 'as'
    lapply(object@groupList, show)
  }
)
# bgVic

is(bmiVic, "bmiGroups")
is(bgVic, "bmiTraj")

bgAna <- new("bmiGroups")
as(bgAna, "bmiTraj") <- bmiAna
bgAna

setIs(
  class1 = "bmiGroups",
  class2 = "ptGroup",
  coerce = function(from, to) {

    numberGroups <- sapply(from@groupList, getNbGroups)
    Smallest <- which.min(numberGroups)
    to <- new("ptGroup")
    to@nbGroups <- getNbGroups(from@groupList[[Smallest]])
    to@group <- getGroup(from@groupList[[Smallest]])
    return(to)
  }
)
is(bgVic, "ptGroup")
as(bgVic, "ptGroup")

setIs(class1 = "bmiGroups", class2 = "ptGroup",
      coerce = function(from, to) {
        numberGroups <- sapply(from@groupList, getNbGroups)
        Smallest <- which.min(numberGroups)
        to <- new("ptGroup")
        to@nbGroups <- getNbGroups(from@groupList[[Smallest]])
        to@group <- getGroup(from@groupList[[Smallest]])
        return(to)
      },
      replace = function(from, value) {
        numberGroups <- sapply(from@groupList, getNbGroups)
        Smallest <- which.min(numberGroups)
        from@groupList[[Smallest]] <- value
        return(from)
      }
)
as(bgVic, "ptGroup")

as(bgVic, "ptGroup") <- groupVic2
# bgVic

setClass(
         Class = "ptGroupFather",
         slots = list(nbGroups = "numeric"),
         contains = "VIRTUAL"
)
new("ptGroupFather")

setClass(Class = "ptGroupSimple",
         slots = list(part = "factor"),
         contains = "ptGroupFather"
)
setClass(Class = "ptGroupEval",
         slots = list(part = "ordered"),
         contains = "ptGroupFather"
)

setGeneric("nbMultTwo", function(object) {standardGeneric("nbMultTwo")})
setMethod("nbMultTwo", "ptGroupFather", function(object) {object@nbGroups <- object@nbGroups*2;   return(object)})
a <- new("ptGroupSimple", nbGroups = 3, part = factor(LETTERS[c(1, 2, 3, 2, 2, 1)]))
nbMultTwo(a)
# b <- new("ptGroupEval", nbGroups = 5, part = ordered(LETTERS[c(1, 5, 3, 4, 2, 4)]))
# nbMultTwo(b)

student <- setRefClass("student")
student$new()

student <- setRefClass("student",
  fields = list(Age = "numeric"))

Bob <- student$new(Age = 11)
cat("Bob is", Bob$Age, "year old. \n")

Bob$Age <- 12
cat("Bob is", Bob$Age, "year old. \n")

Bob$Age <- 11
Mary <- Bob
Mary$Age <- 20
cat("Mary' is", Mary$Age, "year old. \n")
cat("Bob is", Bob$Age, "year old. \n")

Bob$Age <- 11
Mary <- Bob$copy()
Mary$Age <- 20
cat("Mary' is", Mary$Age, "year old. \n")
cat("Bob is", Bob$Age, "year old. \n")

student <- setRefClass("student",
  fields = list(Age = "numeric"),
  methods = list(
    grow = function(x = 1) {
      Age <<- Age + x
    },
    setAge = function(x) {
      Age <<- x
    }
  )
)
Bob <- student$new(Age = 11)
Bob$grow()
cat("Bob is", Bob$Age, "year old. \n")
Bob$setAge(11)
cat("Bob is", Bob$Age, "year old. \n")

studentPlus <- setRefClass("studentPlus", 
                          contains = "student",
                          methods = list(
                              setAge = function(x) {
                                  if(x < 0) stop("Age can't be under 0. ")
                                  Age <<- x
                              }
                          ))

Bob <- studentPlus$new(Age = 11)
Bob$grow(2)
cat("Bob is", Bob$Age, "year old. \n")

Bob$setAge(-1)


