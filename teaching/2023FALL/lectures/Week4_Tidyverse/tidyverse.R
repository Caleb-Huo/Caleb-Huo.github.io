#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Thursday September 7, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: Data manipulation (Tidyverse)
#' ---
#' 
#' 
#' 
#' 
#' Outline
#' ===
#' 
#' The tidyverse is a collection of R packages designed for data science.
#' 
#' - readr
#' - dplyr
#' - tidyr
#' - stringr
#' - ggplot2 (Next week)
#' - others ...
#' 
#' ---
#' 
#' - load these packages individually
#' 
## ---------------------------------------------------------------------------------------------
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)

#' 
#' ---
#' 
#' - alternatively, you can do library(tidyverse) to include all of them
#' 
## ---------------------------------------------------------------------------------------------
library(tidyverse)

#' 
#' 
#' Read in data (readr)
#' ===
#' 
#' - data source: sleepstudy.csv (also available in R lme4 package)
#' 
#' - Original way to read in data
## ---------------------------------------------------------------------------------------------
asleepfile <- "https:///Caleb-Huo.github.io/teaching/data/sleep/sleepstudy.csv"
data0 <- read.csv(asleepfile)

#' 
#' - Use read_csv
#'   - Much faster than read.csv, especially for large datasets
## ---------------------------------------------------------------------------------------------
data1 <- read_csv(asleepfile)

#' 
#' 
#' Inspect the data
#' ===
#' 
#' - read.csv
#'   - You may want to use head, otherwise it will print out everyting
## ---------------------------------------------------------------------------------------------
head(data0)
class(data0)

#' 
#' --- 
#' 
#' - read_csv
#'   - smartly print out the first few rows
## ---------------------------------------------------------------------------------------------
data1
class(data1)

#' 
#' read_delim
#' ===
#' 
#' - Similar to read_csv, we can use read_delim, which is more general
#'   - read_csv assumes the delimiter is ","
#'   - for read_delim, you need to specify the delimiter
#'     - ",": comma delimited (usually for .csv)
#'     - "\\t": tab delimited (usually for .txt)
#'     - " ": space delimited
#' 
## ---------------------------------------------------------------------------------------------
data2 <- read_delim("sleepstudy.csv", delim=",")

#' 
#' Read excel
#' ===
#' 
#' - readxl package
#' - readxl will be included as part of tidyverse
#' - It seems we need to download the dataset to local first
#' 
## ---- eval = FALSE----------------------------------------------------------------------------
## library(readxl)
## ## bsleepfile <- "https:///Caleb-Huo.github.io/teaching/data/sleep/sleepstudy.xlsx"
## data0 <- read_excel("sleepstudy.xlsx")
## data0

#' 
#' - xlsx package
#' 
## ---- eval = FALSE----------------------------------------------------------------------------
## library(xlsx)
## ## bsleepfile <- "https:///Caleb-Huo.github.io/teaching/data/sleep/sleepstudy.xlsx"
## data0 <- read.xlsx("sleepstudy.xlsx", sheetIndex = 1)
## data0

#' 
#' 
#' Write excel
#' ===
#' 
## ---- eval = FALSE----------------------------------------------------------------------------
## library(xlsx)
## ## bsleepfile <- "https:///Caleb-Huo.github.io/teaching/data/sleep/sleepstudy.xlsx"
## data_iris <- iris
## data_cars <- cars
## write.xlsx(data_iris, file = "mydata.xlsx", sheetName="iris")
## write.xlsx(data_cars, file = "mydata.xlsx", sheetName="cars", append = TRUE)

#' 
#' 
#' Read SAS, SPSS, and Stata files.
#' ===
#' 
#' - haven package
#' - haven will be included as part of tidyverse
#' 
## ---------------------------------------------------------------------------------------------
library(haven)
salesfile <- "https:///Caleb-Huo.github.io/teaching/data/sleep/sales.sas7bdat"
data0 <- read_sas(salesfile)
data0

#' 
#' 
#' 
#' 
#' Inspection on data1
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1_sub <- data1[1:3,]
data1_sub$Reaction
as.matrix(data1_sub)
as.data.frame(data1_sub)

#' 
#' 
#' dplyr
#' ===
#' 
#' - dplyr provides a set of efficient tools for data manipulation in R
#' - The key components of dplyr were written in Rcpp, which is very fast and efficient.
#' - naming convention
#'   - precursor of dplyr was plyr
#'   - plyr comes from various "apply" functions in R 
#'   - d is for dataframe
#' 
#' 
#' 
#' dplyr
#' ===
#' 
#' - %>%: pipe
#' - select():	select columns
#' - filter():	filter rows by logical variable
#' - pull():	obtain a specific column
#' - slice():	subset rows by index
#' - arrange():	re-order or arrange rows
#' - mutate():	create new columns
#' - mutate_at():	directly change original data
#' - rename(): rename
#' - summarise():	summarise values
#' - group_by():	allows for group operations in the “split-apply-combine” concept
#' - merge data.frame
#'   - inner_join()
#'   - left_join()
#'   - right_join()
#'   - full_join()
#'   - anti_join()
#' 
#' select
#' ===
#' 
## ---------------------------------------------------------------------------------------------
select(data1, Days, Subject)

#' 
#' ---
#' 
#' - Use pipe
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% select(Days, Subject)

#' 
#' pipe
#' ===
#' 
#' - %>%
#' - data1 %>% select(Days, Subject)
#'   - the result before %>% will be piped into the first argument of the function behind %>%
#' - data1 %>% select(., Days, Subject)
#' - shortcut: 
#'   - cmd + shift + M: MAC
#'   - ctrl + shift + M: Windows
#' 
## ---------------------------------------------------------------------------------------------
exp(1)
1 %>% exp()
1 %>% exp() %>% log ## () can be omitted if the data is the only argument

#' 
#' 
#' ---
#' 
#' - Do not select
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% select(-Reaction)

#' 
#' 
#' 
#' 
#' 
#' filter
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  select(Days, Subject) %>%
  filter(Subject == 308)

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  filter(Reaction >= 300) %>%
  select(Days, Subject) %>%
  filter(Subject == 308)

#' 
#' ---
#' 
#' - multiple filtering criteria
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  filter(Reaction >= 300, Subject == 308) 

data1 %>% 
  filter(Reaction >= 300 & Subject == 308) 

#' 
#' pull
#' ===
## ---------------------------------------------------------------------------------------------
(data1 %>% 
  filter(Reaction >= 300, Subject == 308))$Reaction

data1 %>% 
  filter(Reaction >= 300, Subject == 308) %>%
  pull(Reaction)


#' 
#' slice
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  slice(1:8) 

data1 %>% 
  head(n=8) 

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
which(data1$Reaction >= 300 & data1$Subject == 308)
with(data1, which(Reaction >= 300 & Subject == 308)) ## works for data.frame

data1 %>% 
  slice(which(Reaction >= 300 & Subject == 308)) 

#' 
#' arrange
#' ===
#' 
#' - arrange() is similar to sort() and order()
#' - assending order by default
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% arrange(Reaction) %>% head

#' 
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  arrange(Days, Reaction) %>% 
  head

#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  arrange(Days, Reaction) %>% 
  head %>%
  colSums ## pipe also work for other functions

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  arrange(desc(Days), Reaction) %>% 
  head

#'     
#' 
#' mutate
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  mutate(Reaction_binary = Reaction<250) %>%
  head

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  mutate(Reaction_binary = Reaction<250,
        Reaction_sec = Reaction/1000) %>% 
  head

#' 
#' mutate at (apply a function to one or several columns)
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
  head

data1 %>% 
  head %>%
  mutate_at(c("Reaction", "Subject"), log)


#' 
#' rename
#' ===
## ---------------------------------------------------------------------------------------------
data1 %>% 
  rename(ID = Subject)

#' 
#' summarise
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1 %>% 
    summarise(avg_reaction = mean(Reaction), 
              min_reaction = min(Reaction),
              max_reaction = max(Reaction),
              total = n())

#' 
#' 
## ---------------------------------------------------------------------------------------------
adata <- data1 %>% 
    summarise(avg_reaction = mean(Reaction), 
              min_reaction = min(Reaction),
              max_reaction = max(Reaction),
              total = n())

#' 
#' group_by
#' ===
#' 
#' - seems to be more powerful than tapply
#' 
## ---------------------------------------------------------------------------------------------
tt <- data1 %>% 
      group_by(Subject) %>%
      summarise(avg_reaction = mean(Reaction), 
              min_reaction = min(Reaction),
              max_reaction = max(Reaction),
              total = n())
tt %>% head

#' 
#' 
#' select (2)
#' ===
#' 
#' - Select a range
#' 
## ---------------------------------------------------------------------------------------------
tt %>% 
  head %>% 
  select(avg_reaction:max_reaction)

#' 
#' 
#' ---
#' 
#' - contains
#' 
## ---------------------------------------------------------------------------------------------
tt %>% 
  head %>% 
  select(contains("reaction"))

#' 
#' more options for select()
#' ===
#' 
#' - starts_with() = Select columns that start with a character string
#' - ends_with() = Select columns that end with a character string
#' - contains() = Select columns that contain a character string
#' - matches() = Select columns that match a regular expression
#' - one_of() = Select columns names that are from a group of names
#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
tt %>% 
  head %>% 
  select(ends_with("reaction"))

tt %>% 
  head %>% 
  select(starts_with("m"))


#' 
#' Select by variables that are contained in a character vector
#' ===
#' 
#' - all_of() is for strict selection. 
#' 
## ---------------------------------------------------------------------------------------------
avar <- c("Reaction", "Days")
data1 %>% select(all_of(avar))

#' 
#' - any_of() doesn't check for missing variables. 
#' 
## ---------------------------------------------------------------------------------------------
bvar <- c("Reaction", "Days", "Months")
data1 %>% select(any_of(bvar))

#' 
#' 
#' create a tibble
#' ===
#' 
#' - tibble is similar to a dataframe, but tibble is better designed
#' 
## ---------------------------------------------------------------------------------------------
atibble <- tibble(A = 4:6, B = c("A", "B", "C"))
atibble

#' 
#' 
## ---------------------------------------------------------------------------------------------
adataframe <- data.frame(A = 4:6, B = c("A", "B", "C"))
as_tibble(adataframe)

#' 
#' 
#' - add a index column
## ---------------------------------------------------------------------------------------------
btibble <- atibble %>% rowid_to_column()
btibble

#'   
#' 
#' merge data.frame
#' ===
#' 
#' ![](../figure/join.png){width=80%}
#' 
#' 
#' the data for merge
#' ===
#' 
## ---------------------------------------------------------------------------------------------
superheroes <- "
    name, alignment, gender,         publisher
 Magneto,       bad,   male,            Marvel
   Storm,      good, female,            Marvel
Mystique,       bad, female,            Marvel
  Batman,      good,   male,                DC
   Joker,       bad,   male,                DC
Catwoman,       bad, female,                DC
 Hellboy,      good,   male, Dark Horse Comics
"
superheroes <- read_csv(superheroes, skip = 1)

publishers <- "
  publisher, yr_founded
         DC,       1934
     Marvel,       1939
      Image,       1992
"
publishers <- read_csv(publishers, skip = 1)

#' 
#' inner_join
#' ===
#' 
#' - inner_join(x, y): Return all rows from x where there are matching values in y, and all columns from x and y.
#' - If there are multiple matches between x and y, all combination of the matches are returned.
#' 
## ---------------------------------------------------------------------------------------------
inner_join(superheroes, publishers)

#' 
#' ---
#' 
#' 
## ---------------------------------------------------------------------------------------------
inner_join(publishers, superheroes)

#' 
#' 
#' left_join
#' ===
#' 
#' - left_join(x, y): Return all rows from x, and all columns from x and y. 
#' - If there are multiple matches between x and y, all combination of the matches are returned. 
#' 
## ---------------------------------------------------------------------------------------------
left_join(superheroes, publishers)

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
left_join(publishers, superheroes)

#' 
#' right_join
#' ===
#' 
#' - right_join(x, y): Return all rows from y, and all columns from x and y. 
#' - If there are multiple matches between x and y, all combination of the matches are returned. 
#' 
## ---------------------------------------------------------------------------------------------
right_join(superheroes, publishers)

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
right_join(publishers, superheroes)

#' 
#' 
#' anti_join
#' ===
#' 
#' - anti_join(x, y): Return all rows from x where there are not matching values in y, keeping just columns from x. 
#' 
## ---------------------------------------------------------------------------------------------
anti_join(superheroes, publishers)

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
anti_join(publishers, superheroes)

#' 
#' full_join
#' ===
#' 
#' - full_join(x, y): Return all rows and all columns from both x and y. 
#' - Where there are not matching values, returns NA for the one missing. 
#' 
## ---------------------------------------------------------------------------------------------
full_join(superheroes, publishers)

#' 
#' ---
#' 
## ---------------------------------------------------------------------------------------------
full_join(publishers, superheroes)

#' 
#' 
#' tidyr
#' ===
#' 
#' - gather
#' - separate
#' - spread
#' - merge
#' 
#' 
#' spread
#' ===
#' 
#' - Function:       
#'   - spread(data, key, value, fill = NA)
#' - Same as:        
#'   - data %>% spread(key, value, fill = NA)
#' 
#' - Arguments:
#'   - data:           data frame
#'   - key:            column values to convert to multiple columns
#'   - value:          single column values to convert to multiple columns' values 
#'   - fill:           If there isn't a value for every combination of the other variables and the key column, this value will be substituted
#'   
#' 
#' spread example
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1_wide <- data1 %>% spread(Days, Reaction)
head(data1_wide)

#' 
#' gather
#' ===
#' 
#' - Function: 
#'   - gather(data, key, value, ..., na.rm = FALSE)
#' - Same as: 
#'   - data %>% gather(key, value, ..., na.rm = FALSE)
#' - Arguments:
#'   - data: data frame
#'   - key: column name representing new variable
#'   - value: column name representing variable values
#'   - ...: names of columns to gather (or not gather)
#'   - na.rm: option to remove observations with missing values (represented by NAs)
#'         
#' gather example
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1_long <- data1_wide %>% gather(ddays, rreaction, "0":"9")
head(data1_long)

#' 
#' unite
#' ===
#' 
#' - Function:       
#'   - unite(data, col, ..., sep = " ", remove = TRUE)
#' - Same as:        
#'   - data %>% unite(col, ..., sep = " ", remove = TRUE)
#' - Arguments:
#'   - data: data frame
#'   - col: column name of new "merged" column
#'   - ...: names of columns to merge
#'   - sep: separator to use between merged values
#'   - remove: if TRUE, remove input column from output data frame
#' 
#' unite example
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1_unite<- data1 %>% unite(Subject_Days, Subject, Days, sep="_")
head(data1_unite)

#' 
## ---- eval = FALSE----------------------------------------------------------------------------
## data1_unite<- data1 %>% unite(Subject_Days, Subject, Days, sep="_", remove = FALSE) ## if you want to keep a copy of the original data
## head(data1_unite)

#' 
#' separate
#' ===
#' 
#' - Function:
#'   - separate(data, col, into, sep = " ", remove = TRUE)
#' - Same as:
#'   - data %>% separate(col, into, sep = " ", remove = TRUE)
#' - Arguments:
#'   - data: data frame
#'   - col: column name representing current variable
#'   - into: names of variables representing new variables
#'   - sep: how to separate current variable (char, num, or symbol)
#'   - remove: if TRUE, remove input column from output data frame
#'                         
#' separate example
#' ===
#' 
## ---------------------------------------------------------------------------------------------
data1_separate<- data1_unite %>% separate(Subject_Days, c("subjects", "days"), sep="_")
head(data1_separate)

#'  
## ---- eval = FALSE----------------------------------------------------------------------------
## data1_separate<- data1_unite %>% separate(Subject_Days, c("subjects", "days"), sep="_", remove = FALSE)
## head(data1_separate)

#' 
#' 
#' stringr 
#' ===
#' 
#' stringr package contains a set of commonly used string manipulation functions.
#' 
#' - Detect Matches
#' - Subset Strings
#' - Manage lengths
#' - Mutate Strings
#' - Join and split
#' 
#' 
#' stringr cheatsheet:
#' - https://github.com/rstudio/cheatsheets/blob/master/strings.pdf
#' 
#' 
#' 
#' Detect Matches (1)
#' ===
#' 
#' - str_detect: return a logical vector to indicate match position
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

#' 
## ---------------------------------------------------------------------------------------------
str_detect(colorVec, "e") ## contains e
str_detect(colorVec, "e$") ## ends with e
str_detect(colorVec, "^e") ## starts with e

#' 
#' 
#' Detect Matches (2)
#' ===
#' 
#' - str_which: return a index vector to indicate match position
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

#' 
## ---------------------------------------------------------------------------------------------
str_which(colorVec, "e") ## contains e
str_which(colorVec, "e$") ## ends with e
str_which(colorVec, "^e") ## starts with e

#' 
#' 
#' Detect Matches (3)
#' ===
#' 
#' - str_count: count frequency of match
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

#' 
## ---------------------------------------------------------------------------------------------
str_count(colorVec, "e") ## contains e
str_count(colorVec, "e$") ## ends with e
str_count(colorVec, "^e") ## starts with e

#' 
#' 
#' Subset Strings (1)
#' ===
#' 
#' - str_sub(string, start = 1L, end = -1L): subset of a string
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

#' 
## ---------------------------------------------------------------------------------------------
str_sub(colorVec,2,2)
str_sub(colorVec,start = 1, end=-2)
str_sub(colorVec,end=-2)
str_sub(colorVec,start = 2, end = -1)
str_sub(colorVec,start = 2)

#' 
#' 
#' Subset Strings (2)
#' ===
#' 
#' - str_subset(string, pattern), return the matched string
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

#' 
## ---------------------------------------------------------------------------------------------
str_subset(colorVec,"e")

str_subset(colorVec,"e$")

str_subset(colorVec,"^e")

#' 
#' 
#' Subset Strings (3)
#' ===
#' 
#' - str_extract(string, pattern), extract matching patterns from a string
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

#' 
## ---------------------------------------------------------------------------------------------
str_extract(colorVec,"e")
str_extract(colorVec,"[aeiou]") ## the first match
str_extract_all(colorVec,"[aeiou]") ## all matches

#' 
#' 
#' 
#' Manage lengths 
#' ===
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

str_length(colorVec)
matrix(str_pad(colorVec, width = 7),ncol=1)


#' 
#' 
#' Mutate Strings
#' ===
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")
str_sub(colorVec,1,1) <- "Z" ## will change the original string vector
colorVec

#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")
str_replace(colorVec, "e", "E") 
str_replace_all(colorVec, "e", "E")

str_to_lower(colorVec)
str_to_upper(colorVec)
str_to_title(colorVec) ## like a sentence

#' 
#' 
#' 
#' Join and split
#' ===
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")


str_c(colorVec, seq_along(colorVec))
str_c(colorVec, collapse = "::")

#' 
#' 
#' Order Strings
#' ===
#' 
## ---------------------------------------------------------------------------------------------
colorVec <- c("red", "blue", "green", "yellow", "black", "orange", "purple", "white")

str_order(colorVec) ## same as order(colorVec)
str_sort(colorVec) ## same as sort(colorVec)

#' 
#' 
#' references
#' ===
#' 
#' - http://genomicsclass.github.io/book/pages/dplyr_tutorial.html
#' - https://uc-r.github.io/tidyr
#' - http://stat545.com/bit001_dplyr-cheatsheet.html#inner_joinsuperheroes-publishers
#' - http://www.stat.cmu.edu/~ryantibs/statcomp-F18/
#' - https://stringr.tidyverse.org
