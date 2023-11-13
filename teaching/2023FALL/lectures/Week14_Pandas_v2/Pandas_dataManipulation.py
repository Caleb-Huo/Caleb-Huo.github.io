#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Nov 14th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Pandas data manipulation"
#' ---
#' 
## ----setup, include=FALSE---------------------------------------
library(reticulate)
#use_python("/usr/local/bin/python3.10")
use_python("/Users/zhuo/anaconda3/envs/py311/bin/python")

#' 
## import pandas as pd

## import numpy as np

#' 
#' Outlines
#' ===
#' 
#' - Hierarchical Indexing
#' - Merge and Combine
#'   - merge on common columns
#'   - merge on index
#' - Reshaping and Pivoting
#' - GroupBy
#' - Data Aggregation
#' 
#' 
#' Hierarchical Indexing
#' ===
#' 
#' - The hierarchical indexing enables two or more index level on axis
#' 
## data = pd.Series(np.arange(6), index = [["l1", "l1", "l1", "l2", "l2", "l2"], list("abc") * 2])

## data

## data.index

#' 
#' ---
#' 
#' - selection by index
#' 
## data["l1"]

## data.loc["l2"]

## data.loc["l2", "b"]

#' 
#' - selection by inner index
#' 
## data.loc[:,"a"]

#' 
#' ---
#' 
#' - transform a multiIndex Series to a DataFrame
#' 
## data.unstack()

#' 
#' - transform a DataFrame to a multiIndex Series
#' 
## data.unstack().stack()

#' 
#' Swap index
#' ===
#' 
## data = pd.Series(np.arange(6), index = [["l1", "l1", "l1", "l2", "l2", "l2"], list("abc") * 2])

## data.index.names = ["Key1", "Key2"]

## data.swaplevel("Key1", "Key2")

## data.swaplevel(0, 1).sort_index()

#' 
#' 
#' 
#' Hierarchical Indexing
#' ===
#' 
#' - Hierarchical indexing for both row index and column names
#' 
## data2 = pd.DataFrame(np.arange(16).reshape((4, -1)),

##     index=[["l1", "l1", "l2", "l2"], ["a", "b", "a", "b"]],

##     columns=[["TX", "TX", "FL", "FL"],["Red", "Blue", "Red",  "Blue"]])

## data2.index.names = ["Key1", "Key2"]

## data2.columns.names = ["State", "Color"]

## data2

## data2["FL"]

#' 
#' get indexes
#' ===
#' 
#' - row index
#' 
## data2.index.get_level_values(0)

## data2.index.get_level_values("Key1")

## data2.index.get_level_values(1)

#' 
#' - column index
#' 
## data2.columns.get_level_values(0)

## data2.columns.get_level_values("State")

## data2.columns.get_level_values(1)

#' 
#' Drop levels
#' ===
#' 
#' 
## data2.index.droplevel()

## data2.index.droplevel(0)

## data2.index.droplevel(1)

## 

## data2.columns.droplevel()

## data2.columns.droplevel(0)

## data2.columns.droplevel(1)

#' 
#' 
#' Creating index with DataFrame's columns
#' ===
#' 
#' 
## adata = pd.DataFrame({"c1": [1,1,1,2,2,2,2],

##         "c2": list("abcdefg"),

##         "c3": range(7,0,-1),

##         "c4": list("abababa")}

## )

## 

## adata.set_index("c1")

## adata.set_index(["c1"], drop=False)

#' 
#' ---
#' 
## bdata = adata.set_index(["c1", "c4"])

## bdata

## bdata.reset_index()

#' 
#' 
#' merge based on common column
#' ===
#' 
#' - common column in this example: key
#'     - by default, pd.merge will search for the common column
#' 
## data1 = pd.DataFrame({"key": list("bccaab"), "value1": range(6)})

## data2 = pd.DataFrame({"key": list("abc"), "value2": range(3)})

## pd.merge(data1, data2)

#' 
#' - specify the common column explicitly
#' 
## pd.merge(data1, data2, on = "key")

#' 
#' 
#' merge based on common column
#' ===
#' 
#' - if the two columns (column to merge) has different values, we can specify the "how" method
#'     - inner: observations existed in both tables (default)
#'     - left: observations in the left table
#'     - right: observations in the right table
#'     - outer: observations existed in either tables
#' 
## data1 = pd.DataFrame({"key": list("bccaab"), "value1": range(6)})

## data2 = pd.DataFrame({"key": list("abd"), "value2": range(3)})

## pd.merge(data1, data2, how="inner") ## default

## pd.merge(data1, data2, how="left")

#' 
#' ---
#' 
## pd.merge(data1, data2, how="right")

## pd.merge(data1, data2, how="outer")

#' 
#' 
#' merge based on common column
#' ===
#' 
#' - the common column may have different names in two columns
#' 
## data1 = pd.DataFrame({"key1": list("bccaab"), "value1": range(6)})

## data2 = pd.DataFrame({"key2": list("abc"), "value2": range(3)})

## pd.merge(data1, data2, left_on="key1", right_on="key2")

#' 
#' 
#' 
#' 
#' merge based on index
#' ===
#' 
#' - based on indexes from both dataFrame
#' 
## data1 = pd.DataFrame({"key": list("bccaab"), "value1": range(6)})

## data2 = pd.DataFrame({"key": list("abd"), "value2": range(3)})

## data1_key = data1.set_index("key")

## data2_key = data2.set_index("key")

## pd.merge(data1_key,data2_key, left_index=True, right_index=True)

#' 
#' - based on index from one dataFrame, and a column from another dataFrame
#' 
## pd.merge(data1,data2_key, left_on="key", right_index=True)

#' 
#' 
#' 
#' merge using the join method
#' ===
#' 
#' 
## data1 = pd.DataFrame({"key": list("bccaab"), "value1": range(6)})

## data2 = pd.DataFrame({"key": list("abd"), "value2": range(3)})

## data1_key = data1.set_index("key")

## data2_key = data2.set_index("key")

#' 
#' - join method: allows to perform left join
#'     - both DataFrame have the same index
#' 
## data2_key.join(data1_key)

#' 
#' ---
#' 
#' - swap arguments based on the previous example
#' 
## data1_key.join(data2_key)

#' 
#' - join method:
#'     - left dataFrame uses a key column
#'     - right dataFrame uses the index
#' 
## data1.join(data2_key, on="key")

#' 
#' 
#' 
#' Concatenating 
#' ===
#' 
#' - pd.Series
#'     - concatenate rows
#' 
## series1 = pd.Series([1,2,3], index=list("abc"))

## series2 = pd.Series([3,5,7], index=list("abd"))

## 

## pd.concat([series1, series2])

## res = pd.concat([series1, series2], keys=["data1", "data2"])

## res

## res.unstack()

#' 
#' ---
#' 
#' - pd.Series
#'     - concatenate columns
#' 
## pd.concat([series1, series2], axis=1)

## pd.concat([series1, series2], axis=1, join="inner")

#' 
#' 
#' Concatenating 
#' ===
#' 
#' - pd.DataFrame
#'     - concatenate rows
#' 
## df1 = pd.DataFrame(np.arange(6).reshape(3,-1), index=list("ABC"), columns= ["C1", "C2"])

## df2 = pd.DataFrame(np.arange(4).reshape(2,-1), index=list("AB"), columns= ["C2", "C3"])

## 

## pd.concat([df1, df2])

## pd.concat([df1, df2], ignore_index=True)

#' 
#' ---
#' 
#' - pd.DataFrame
#'     - concatenate columns
#' 
## df1 = pd.DataFrame(np.arange(6).reshape(3,-1), index=list("ABC"), columns= ["C1", "C2"])

## df2 = pd.DataFrame(np.arange(4).reshape(2,-1), index=list("AB"), columns= ["D1", "D2"])

## 

## pd.concat([df1, df2], axis=1)

## pd.concat([df1, df2], axis=1, keys=["Level1", "Level2"])

## pd.concat({"Level1": df1, "Level2": df2}, axis=1)

#' 
#' 
#' 
#' Combining data with overlaps 
#' ===
#' 
#' - pd.Series (s1 and s2 have overlaps)
#'     - return the elements in s1 first
#'     - if s1 contains missing values, return the elements in s2
#'     
## s1 = pd.Series([np.nan,1,2,3,np.nan], index=list("abcde"))

## s2 = pd.Series([1,np.nan,6,7, np.nan], index=list("bcdea"))

## 

## np.where(pd.isna(s1), s2, s1) ## this will ignore index

## s1.combine_first(s2) ## also match index

#' 
#' 
#' ---
#' 
#' - pd.DataFrame
#'     - return the elements in pd1 first
#'     - if pd1 contains missing values, return the elements in pd2
#'     
## df1 = pd.DataFrame({"C1": [np.nan, 1, 2],

##                     "C2": [5, np.nan, 3],

##                     "C3": [np.nan, 9, np.nan]

## }, index=list("ABC"))

## df2 = pd.DataFrame({"C1": [1,2,],

##                     "C2": [4,5,]

## }, index=list("AB"))

## 

## df1

## df2

## df1.combine_first(df2)

#' 
#' 
#' 
#' Stack and unstack
#' ===
#' 
#' - stack:
#'     - move the data from the columns to rows
#' 
## pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## pd1.columns.name="Columns"

## pd1.index.name="States"

## pd1

## pd1_stacked = pd1.stack()

## pd1_stacked

#' 
#' 
#' Stack and unstack
#' ===
#' 
#' - unstack:
#'     - re-arrange a hierarchical index Series back to a DataFrame
#'     - can specify the level of index to be unstacked
#'     
## pd1_stacked.unstack()

## pd1_stacked.unstack(0)

#' 
#' ---
#' 
## pd1_stacked.unstack("Columns")

## pd1_stacked.unstack("States")

#' 
#' Specify the levels in both stack and unstack
#' ===
#' 
## pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## pd1.columns.name="Columns"

## pd1.index.name="States"

## # pd1

## pd1.stack("Columns").unstack("States")

#' 
#' - this is the same as transpose
#' 
## pd1.T

#' 
#' 
#' Pivoting between long and wide format
#' ===
#' 
#' - long format to wide format
#' - wide format to long format
#' - example data
#' 
## url="https://caleb-huo.github.io/teaching/data/sleep/sleepstudy.csv"

## sleepdata=pd.read_csv(url)

## sleepdata.head()

#' 
#' 
#' Pivoting from long format to wide format
#' ===
#' 
#' - pivot
#'   - index: Column to use to make new frame’s index
#'   - columns: Column to use to make new frame’s columns
#' 
## sleepdata_wide = sleepdata.pivot(index="Subject", columns="Days")

## sleepdata_wide.head()

#' 
#' - alternative approach
#' 
#' ```
#' sleepdata.set_index(["Subject", "Days"]).unstack("Days")
#' ```
#' 
#' 
#' Pivoting from wide format to long format
#' ===
#' 
#' - melt (toy example)
#'   - id_vars: Column(s) to use as identifier variables
#'   - value_vars: Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars
#'   
## pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## pd1.index.name = "State"

## pd2 = pd1.reset_index()

## pd2

## 

## pd2.melt(id_vars="State", value_vars=["b", "d", "c"])

## 

#' 
#' ---
#' 
#' - melt for the sleeping data
#' 
## sleepdata_wide3 = sleepdata_wide

## sleepdata_wide3.columns = sleepdata_wide.columns.droplevel()

## sleepdata_wide3.reset_index(inplace=True)

## sleepdata_long = pd.melt(sleepdata_wide3, id_vars = ["Subject"], value_vars = range(10))

## sleepdata_long

#' 
#' ---
#' 
#' - using stack() method
#' 
## sleepdata_wide3.set_index("Subject").stack().reset_index()

#' 
#' 
#' Groupby with single group key
#' ===
#' 
#' - single group key
#' 
## adata = pd.DataFrame({

##     "Key1": list("AAABBB"),

##     "value1": np.arange(6),

## })

## adata

#' 
#' - groupby
#' 
## adata.groupby("Key1").mean()

## adata["value1"].groupby(adata["Key1"]).mean() ## this may be more interpretable

#' 
#' Iteration over groups
#' ===
#' 
#' 
## adata = pd.DataFrame({

##     "Key1": list("AAABBB"),

##     "value1": np.arange(6),

## })

## grouped_data = adata.groupby("Key1")

## 

## for name, data in grouped_data:

##     print(name)

##     print(data)

#' 
#' Groupby and dictionary
#' ===
#' 
## adata = pd.DataFrame({

##     "Key1": list("AAABBB"),

##     "value1": np.arange(6),

## })

## grouped_data = adata.groupby("Key1")

## 

## adict = dict(list(grouped_data))

## adict.keys()

## adict["A"]

#' 
#' 
#' Groupby with multiple group key
#' ===
#' 
#' - multiple group keys
#' 
## bdata = pd.DataFrame({

##     "Key1": list("AAABBB"),

##     "Key2": ["blue", "orange", "blue", "orange", "blue", "orange"],

##     "value1": np.arange(6),

##     "value2": np.arange(10,16)

## })

## bdata

#' 
#' - apply functions after groupby
#' 
## bdata.groupby(["Key1", "Key2"]).mean()

## bdata.groupby(["Key1", "Key2"]).size()

#' 
#' ---
#' 
#' - apply functions to a specific column
#' 
## bdata.groupby(["Key1", "Key2"])["value1"].mean()

## bdata.groupby(["Key1", "Key2"])["value2"].size()

#' 
#' 
#' Grouping with functions
#' ===
#' 
#' - length of the index
#' 
## bdata.index = ["Amy", "Beth", "Carl", "Dan", "Emily", "Frank"]

## bdata

## bdata.groupby(len)[["value1", "value2"]].min()

#' 
#' Data Aggregation
#' ===
#' 
#' - agg function: to apply a function
#' 
## bdata.drop(columns = "Key2").groupby("Key1").agg("median")

#' 
#' - user defined function
#'     - maxDiff
#' 
## def maxDiff(arr):

##     return(arr.max() - arr.min())

## 

## bdata.drop(columns = "Key2").groupby("Key1").agg(maxDiff)

## bdata.drop(columns = "Key2").groupby("Key1").agg(lambda x: x.max() - x.min())

#' 
#' Data Aggregation
#' ===
#' 
#' - multiple functions
#' 
## bdata.drop(columns = "Key2").groupby("Key1").agg(["median", "mean", maxDiff])

#' 
#' - set names for the result
#' 
## bdata.drop(columns = "Key2").groupby("Key1").agg([("mymedian","median"), ("mymean", "mean"), ("mymaxDiff", maxDiff)])

#' 
#' Data Aggregation with selected columns
#' ===
#' 
#' - multiple functions with different columns
#' 
## bdata.drop(columns = "Key2").groupby("Key1").agg({"value1": ["median", "mean"], "value2": [maxDiff]})

## bdata.groupby(["Key1", "Key2"]).agg({"value1": ["median", "mean"], "value2": [maxDiff]})

#' 
#' 
#' Data Aggregation without row index
#' ===
#' 
#' - as_index = False
#' 
## bdata.groupby(["Key1", "Key2"], as_index=False).agg("mean")

#' 
#' 
#' Data Aggregation
#' ===
#' 
#' - two ways to calculate mean value
#' 
## bdata.groupby(["Key1", "Key2"], as_index=False).agg("mean")

## bdata.groupby(["Key1", "Key2"], as_index=False).mean()

#' 
#' 
#' Data Aggregation for functions with extra argument
#' ===
#' 
#' 
## def myMean(arr, offset = 5):

##     return(arr.mean() - offset)

## 

## bdata.groupby(["Key1", "Key2"], as_index=False).agg(myMean)

## bdata.groupby(["Key1", "Key2"], as_index=False).agg(myMean, offset=4)

#' 
#' 
#' 
#' Data Aggregation by apply function
#' ===
#' 
#' - apply function is similar to the agg function
#' 
## bdata.groupby(["Key1", "Key2"])["value1"].apply(np.mean)

## bdata.groupby(["Key1", "Key2"], as_index=False)["value1"].apply(np.mean)

## bdata.groupby(["Key1", "Key2"])["value2"].apply(myMean)

## bdata.groupby(["Key1", "Key2"])["value2"].apply(myMean, offset=3)

#' 
#' 
#' Reference
#' ===
#' 
#' - https://wesmckinney.com/book/data-wrangling.html
#' - https://wesmckinney.com/book/data-aggregation.html
#' 
