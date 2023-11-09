#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Nov 9th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Pandas Basics"
#' ---
#' 
## ----setup, include=FALSE---------------------------------------
library(reticulate)
#use_python("/Users/zhuo/opt/miniconda3/envs/py312/bin/python")
use_python("/Users/zhuo/anaconda3/envs/py311/bin/python")

#' 
#' 
#' 
#' Outlines
#' ===
#' 
#' - Introduction to pandas
#' - Series
#' - DataFrame
#' - Pandas basic operations
#' - Function applications
#' - Pandas file IO
#' - TimeSeries
#' - Data cleaning
#' 
#' Introduction to Pandas
#' ===
#' 
#' - Pandas is an open source python package
#'   - data cleaning
#'   - data manipulation
#'   - data preparation
#'     - for statistical modeling
#'     - for data visualization
#' 
#' - import the pandas package
#' 
## import numpy as np

#' 
## import pandas as pd

#' 
#' - if you need to install pandas:
#'   - https://pandas.pydata.org/docs/getting_started/install.html
#' 
#' Series
#' ===
#' 
#' - A Series is one-dimensional object, which contains
#'   - value: array-like values
#'   - index: labels for the values
#' 
## aseries = pd.Series([1,2,3])

## aseries

## aseries.values

## aseries.index ## like range(3)

#' ```
#' [i for i in aseries.index]
#' list(aseries.index)
#' ```
#' 
#' 
#' Series
#' ===
#' 
#' - to create a Series with user-specified index
#' 
## bseries = pd.Series([1.,2.2,3.5], index=['a', 'b', 'c'])

## bseries

## bseries.index ## like range(3)

## ## [i for i in bseries.index]

## ## list(bseries.index)

#' 
#' 
#' Series
#' ===
#' 
#' - Series is like a numpy array
#' 
## bseries*10

## np.log(bseries)

#' 
#' 
#' Series
#' ===
#' 
#' - series is like a dictionary
#' 
## bseries['a']

## bseries.a

## bseries[['a','b']]

## 'a' in bseries

#' 
#' ---
#' 
#' - series can be created via a dictionary
#' 
## adict = {"a":1, "b":2, "c":3}

## pd.Series(adict)

#' 
#' - we can also specify indexes when creating Series
#' 
## cseries = pd.Series(adict, index=["a", "c", "b"])

## cseries

#' 
#' ---
#' 
#' - what if we specify indexes that doesn't exist
#' 
## dseries = pd.Series(adict, index=["d", "c", "b"])

## dseries ## nan will be created

## pd.isnull(dseries)

#' 
#' 
#' Series meta data
#' ===
#' 
#' - meta data for Series
#' 
## bseries.name="XX"

## bseries.index.name="YY"

## ## bseries

#' 
#' 
#' 
#' Series indexing 
#' ===
#' 
#' - index
#' 
## obj = pd.Series(np.arange(3), index=["a", "b", "c"])

## obj

## obj.index ## index is like range()

## obj.index[:-1]

#' 
#' - index is not directly mutable
#' 
#' ```
#' obj.index[0] = "x"
#' ```
#' 
#' - the Series index can be altered in-place by assignment
#' 
## obj.index

## obj.index = ["aa", "bb", "cc"]

## obj

#' 
#' 
#' Reindexing 
#' ===
#' 
#' - Series
#'   - reorder the index
#'   
## data = pd.Series([1.1, 2.2, 3.3], index = ['b', 'c', 'a'])

## data

## data2 = data.reindex(['a', 'b', 'c', 'd'])

## data2

#' 
#' 
#' 
#' 
#' Drop entries in Series
#' ===
#' 
#' - by default, not in place operator (inplace = False)
#' 
## data = pd.Series(np.arange(4), index = ['a', 'b', 'c', 'd'])

## data

## data.drop("a") ## not inplace operator

## data2 = data.drop(["a", "b"])

## data2

## data.drop(["a"], inplace=True)

## data

#' 
#' 
#' Series entry selection
#' ===
#' 
#' - by index
#' 
## data = pd.Series([1,2,3,4], index=["a", "b", "c", "d"])

## data["b"]

#' 
#' 
#' - by row ID
#' 
#' 
## data[1]

## data[[1]]

## data[1:3]

#' 
#' 
#' Series entry selection (more)
#' ===
#' 
## data[[1,3]]

## data[data > 2] ## by logical array

## data["b":"c"]

## ##data["b":"c"] = 0

#' 
#' Series entry selection by loc/iloc
#' ===
#' 
## data = pd.Series([1,2,3,4], index=["a", "b", "c", "d"])

#' 
#' - loc: by index name
#' 
## data.loc["a"]

## data.loc[["a", "c"]]

#' 
#' ---
#' 
#' - iloc: by row ID
#' 
## data.iloc[1]

## data.iloc[1:]

## data.iloc[-2]

#' 
#' Series membership
#' ===
#' 
## data = pd.Series(list("abcdadcccdaabbb"))

#' 
#' - unique
#' 
## data.unique()

#' 
#' - value counts
#' 
## data.value_counts()

#' 
## pd.Series.value_counts(data, sort = False)

#' 
#' Series membership
#' ===
#' 
#' - membership
#' 
## mask = data.isin(["a", "b"])

## data[mask]

#' 
## uniq_val = data.unique()

## pd.Index(uniq_val).get_indexer(data)

#' 
#' Arithmetic and data alignment
#' ===
#' 
#' 
## a1 = pd.Series([1,2,3], index = ["a", "b", "c"])

## a2 = pd.Series([2,3,4], index = ["a", "b", "c"])

## a3 = pd.Series([2,3,4], index = ["b", "c", "d"])

#' 
#' - if indexes are the same
#' 
## a1 + a2

#' 
#' - if indexes are different
#' 
## a1 + a3

#' 
#' 
#' DataFrame
#' ===
#' 
#' - DataFrame
#'   - a dict of Series, which all share the same index
#'   - index on rows, Series on columns
#'   - different columns (Series) can be of different data types
#' 
## data = {"Name": ["Amy", "Beth", "Carl"],

##         "Age": [24, 22, 19],

##         "Sex": ["F", "F", "M"]

## }

## data

## pd.DataFrame(data)

#' 
#' ---
#' 
#' - reorder the columns
#' 
## pd.DataFrame(data, columns = ["Sex", "Age", "Name"])

#' 
#' - if a column name doesn't exist, missing value will be created
#' 
## pd.DataFrame(data, columns = ["Sex", "Age", "Name", "Email"])

#' 
#' - if there was no user-defined index, the index argument will create indexes
#' 
## apd = pd.DataFrame(data, columns = ["Sex", "Age", "Name"], index = ["a", "b", "c"])

## apd

#' 
#' 
#' reset_index and set_index
#' ===
#' 
## apd

## data2 = apd.reset_index()

## data2

## data2.set_index("index")

## #data2.set_index("Name")

#' 
#' 
#' DataFrame - select columns
#' ===
#' 
#' - select columns
#' 
## apd["Age"]

## apd.Age

## apd[["Name", "Age"]]

#' 
#' DataFrame - select rows
#' ===
#' 
#' - by row index
#' 
## ## apd[1] # this won't work

## apd[1:]

## apd[apd["Age"] > 20]

#' 
#' Selection with loc
#' ===
#' 
#' - loc: by row/column name
#' 
## apd.loc["a"]

## apd.loc[:,"Name"]

## apd.loc["a", ["Name", "Age"]]

#' 
#' 
#' Selection with iloc
#' ===
#' 
#' - iloc: by row/column integer index
#' 
## apd.iloc[0]

## apd.iloc[-1]

## apd.iloc[:,1]

## apd.iloc[0, [0,1]]

#' 
#' 
#' create columns
#' ===
#' 
#' 
## apd["Age20"] = apd.Age >= 20

## apd

## apd["debt"] = np.nan

## apd

#' 
#' 
#' modify columns
#' ===
#' 
## apd["debt"] = 15.1

## apd

## apd["debt"] = np.arange(3.)

## apd

#' 
#' - if the assigned value is a Series, only the element with existing indexes will be assigned
#' 
## val = pd.Series([5., 7.7], index=["a", "c"])

## apd["debt"] = val

## apd

#' 
#' 
#' more on columns for pd.DataFrame
#' ===
#' 
#' - get column names
#' 
## apd.columns

#' 
#' - delete columns
#' 
## del apd["debt"]

## apd.columns

#' 
#' - transpose the DataFrame
#' 
## apd.T

#' 
#' 
#' create DataFrame via nested dictionary
#' ===
#' 
#' - when pass a nested dictionary to pd.DataFrame:
#'     - outer dict keys will be columns
#'     - inner dict keys will be row index
#' 
## pop = {"Florida": {2020: 10.9, 2021: 11.3, 2022: 13.4},

##         "Texas": {2020: 20.5, 2021: 21.1}

## }

## bpd = pd.DataFrame(pop)

## bpd

#' 
#' 
#' some extra operations on pd.DataFrame
#' ===
#' 
#' 
#' - set meta data
#' 
## bpd.index.name = "Year"

## bpd.columns.name = "States"

## bpd

#' 
#' 
#' some attributes of pd.DataFrame
#' ===
#' 
#' - values attribute: return numpy array
#' 
## bpd.values

#' 
#' - index attribute
#' 
## bpd.index

## 2021 in bpd.index

#' 
#' - columns attribute
#' 
## bpd.columns

## "Florida" in bpd.columns

#' 
#' 
#' 
#' 
#' 
#' Reindexing for DataFrame
#' ===
#' 
#' - reindexing rows
#' 
## data = pd.DataFrame(np.arange(6).reshape(3,-1), index =['x', 'y', 'z'], columns = ["Florida", 'Texas'] )

## data

## data2 = data.reindex(['x', 'y', 'z', 'w'])

## data2

#' 
#' ---
#' 
#' - reindexing columns
#' 
## states = ['Utah', 'Florida', 'Texas']

## data3 = data2.reindex(columns = states)

## data3

## data3.loc[["x", "y"], states]

#' 
#' 
#' Drop entries in DataFrame
#' ===
#' 
#' - by default, drop index by rows (axis=0)
#' - to drop columns, set axis = 1 or axis = "columns"
#' - by default, not in place operator (inplace = False)
#' 
## data0 = {"Name": ["Amy", "Beth", "Carl"],

##         "Age": [24, 22, 19],

##         "Sex": ["F", "F", "M"]

## }

## data = pd.DataFrame(data0, index = ["1", "2", "3"])

## data.drop("1") ## drop by index

## data.drop("Name", axis=1)

## data.drop(["Name", "Age"], axis="columns")

## 

## ## data.drop("1", inplace=True); data

#' 
#' 
#' Data alignment for DataFrame
#' ===
#' 
#' 
## pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## pd2 = pd.DataFrame(np.arange(12).reshape(-1,3), columns = list("bac"), index = ["Florida", "Texax", "Utah", "Ohio"])

## 

## pd1

## pd2

## pd1 + pd2

#' 
#' 
#' 
#' Arithmetic for DataFrame
#' ===
#' 
## pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## pd3 = pd.DataFrame(np.arange(1,10).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## 

## pd1.add(pd3)

## pd1.sub(1)

#' 
#' 
#' Operation between DataFrame and Series
#' ===
#' 
## pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## 

## series1 = pd1.iloc[0]

## 

## pd1

## 

## series1

## 

## pd1 - series1

#' 
#' Sorting and Ranking
#' ===
#' 
#' - Series
#' 
## series1 = pd.Series([1,3,2,4], index = ['b', 'd', 'a', 'c'])

## series1

## series1.sort_index()

## series1.sort_index(ascending=False)

## series1.sort_values()

#' 
#' 
#' Sorting and Ranking
#' ===
#' 
#' - DataFrame
#' 
## pd1 = pd.DataFrame(np.array([3,2,5,1,4,6]).reshape(3,2), index = ['c', 'a', 'b'], columns = ["x", "y"])

## pd1.sort_index()

## pd1.sort_values(["x"])

## pd1.sort_values(["y", "x"]) ## ## multiple criteria

#' 
#' 
#' 
#' 
#' Function application
#' ===
#' 
## pd1 = pd.DataFrame(np.arange(12).reshape(3,4), index = ['x', 'y', 'z'], columns = ["a", "b", "c", "d"])

## 

## 

## pd1.apply(np.mean)

## pd1.apply(np.mean, axis = 1)

#' 
#' 
#' Function application
#' ===
#' 
#' - lambda function
#' 
## f = lambda x: x.max() - x.min()

## pd1.apply(f)

## pd1.apply(lambda x: x.max() - x.min())

#' 
#' - user specified functions
#' 
## def f2(x):

##     return pd.Series([x.min(), x.max()], index = ["min", "max"])

## 

## pd1.apply(f2)

#' 
#' 
#' 
#' 
#' 
#' Summary statistics
#' ===
#' 
#' 
## pd1.sum() ## default axis = 0

## pd1.sum(axis = 1)

#' 
#' ---
#' 
## pd1.median()

## pd1.idxmax()

#' 
#' Summary statistics
#' ===
#' 
#' 
## pd1.idxmin(axis = "columns")

## pd1.cumsum()

## pd1.var() ## default axis = 0

## ## pd1.std(axis="columns")

#' 
#' 
#' Summary statistics
#' ===
#' 
## pd1.describe()

#' 
#' 
#' 
#' Correlation
#' ===
#' 
#' - correlation: corr()
#' 
## rng = np.random.default_rng(42)

## pd2 = pd.DataFrame(rng.standard_normal(12).reshape(3,4), index = ['x', 'y', 'z'], columns = ["a", "b", "c", "d"])

## 

## pd2["a"].corr(pd2["b"])

## pd2.a.corr(pd2.b)

## pd2.corr()

#' 
#' 
#' Covariance
#' ===
#' 
#' - covariance: cov()
#' 
## pd2["a"].cov(pd2["b"])

## pd2.cov()

#' 
#' Value counts for DataFrame
#' ===
#' 
## data = pd.DataFrame({'a':[1,2,3,3], 'b':[1,1,2,4], 'c':[2,2,3,3]})

## data["a"].value_counts()

## data.apply(pd.Series.value_counts)

## data.apply(pd.Series.value_counts).fillna(0)

#' 
#' 
#' Read in files using Pandas
#' ===
#' 
#' - read_csv: read comma-delimited file, the delimitor can be specified by users
#' - read_excel: read excel files
#' - read_sas: read data in SAS format
#' - read_json: read data in JSON (JavaScript Object Notation) format
#' 
#' 
#' ```
#' afile = "https://caleb-huo.github.io/teaching/data/Python/Student_data.csv"
#' bfile = "https://caleb-huo.github.io/teaching/data/Python/Student_data.xlsx"
#' data0 = pd.read_csv("sleepstudy.csv")
#' data1 = pd.read_excel("sleepstudy.xlsx")
#' ```
#' 
#' Read_csv directly through an URL
#' ===
#' 
## url="https://caleb-huo.github.io/teaching/data/Python/Student_data.csv"

## c=pd.read_csv(url)

## c.head()

#' 
#' 
#' Read_csv more options
#' ===
#' 
#' - skip header
#' 
## data = pd.read_csv("Student_data.csv", header=None)

## data.head()

#' 
#' - skip one line
#' 
## data = pd.read_csv("Student_data.csv", header=None, skiprows = 1)

## data.head()

#' 
#' 
#' Read_csv more options
#' ===
#' 
#' - set column names
#' 
## data = pd.read_csv("Student_data.csv", skiprows = 1, names = ["a", "b", "c", "d", 'e'])

## data.head()

#' 
#' - specify index
#' 
## data = pd.read_csv("Student_data.csv", index_col = "Name")

## data.head()

## data.sort_index().head()

#' 
#' Save results to csv or other format
#' ===
#' 
## data = pd.read_csv("Student_data.csv")

## data.to_csv("mydata.csv") ## index=True as default

## data.to_csv("mydata.csv", index=False)

## data.to_csv("mydata.txt", sep="\t")

#' 
#' - show the results to the console: sys.stdout
#' 
#' ```
#' import sys
#' data.to_csv(sys.stdout) 
#' data.to_csv(sys.stdout, sep="\t") 
#' data.to_csv(sys.stdout, index=False) 
#' data.to_csv(sys.stdout, header=False) 
#' data.to_csv(sys.stdout, columns = ["Hobby", "Year_in_colledge"]) 
#' ```
#' 
#' Review datetime
#' ===
#' 
## from datetime import datetime

## from datetime import timedelta

## 

## now = datetime.now()

## now

## now.year, now.month, now.day

## now.hour, now.minute, now.second

#' 
#' timedelta
#' ===
#' 
## datetime1 = datetime(2022,10,13,7,30,0)

## datetime2 = datetime(2022,10,10,5,20,0)

## delta = datetime1 - datetime2

## delta

## delta.days

## delta.seconds

## 

## datetime1 + timedelta(12)

## datetime1 + timedelta(12, 2)

#' 
#' 
#' 
#' String and Datatime Convertion
#' ===
#' 
#' - datetime to string
#' 
## date1 = datetime(2022,10,13)

## str(date1)

## date1.strftime("%Y-%m-%d")

#' 
#' - string to datetime
#' 
## val1 = "2021-10-11"

## datetime.strptime(val1, "%Y-%m-%d")

## 

## val2 = "10/11/2021"

## datetime.strptime(val2, "%m/%d/%Y")

#' 
#' - string to datetimeIndex
#' 
## dateStr = ["2021-10-11 12:00:01", "2021-10-12 03:40:01"]

## pd.to_datetime(dateStr)

#' 
#' Time Series Basics
#' ===
#' 
#' - Series with datetime as index
#' 
## mydate = [datetime(2022,10,13), datetime(2022,10,14), datetime(2022,10,18)]

## data = pd.Series(np.arange(30,33), index=mydate)

## data

## data.index

#' 
#' - selecting rows with datatimeIndex is flexible
#' 
## data["2022/10/13"]

## data["20221014"]

## data["2022-10-18"]

#' 
#' 
#' Date range
#' ===
#' 
#' - date_range()
#' 
## ind1 = pd.date_range("2022-01-01", "2023-08-01")

## ind1

## 

## ind2 = pd.date_range(start = "2022-01-01", periods=35)

## ind2

#' 
#' 
## ind3 = pd.date_range(end = "2022-01-01", periods=35)

## ind3

## 

## ind4 = pd.date_range(start = "2012-01-01", periods=10, freq="MS") ## MS: month start

## ind4

## 

#' 
#' Date selections
#' ===
#' 
## data = pd.DataFrame(rng.standard_normal(len(ind1)*2).reshape(-1,2), index=ind1, columns=["a", "b"])

## 

## data.head()

## data.loc["2022"] ## all rows in a year

## data.loc["2022-10"] ## all rows in a month

## data.loc["2022-10-12"]

## data.loc["2022-10-12":] ## starting from a date

#' 
#' Missing data -- Series
#' ===
#' 
#' - missing data are denoted as
#'   - np.NaN
#'   - np.nan
#'   - None
#' 
#' 
## vec_withna = pd.Series([0, np.NaN, np.nan, None, 1])

## vec_withna.isnull()

## vec_withna.isna()

## vec_withna[vec_withna.notnull()]

## vec_withna[vec_withna.notna()]

## vec_withna.dropna()

#' 
#' Missing data -- DataDrame
#' ===
#' 
#' - create a new variable NA = np.nan
#' - dropna: by default, drop a row if any missing value exists
#' 
## from numpy import nan as NA

## 

## apd = pd.DataFrame([[1,2,3], [NA, NA, NA], [4, NA, 6]])

## apd.dropna()

#' 
#' - drop if all elements are missing
#' 
## apd.dropna(how="all")

#' 
#' ---
#' 
#' - drop by columns
#' 
## apd[4] = NA

## apd

## apd.dropna(axis=1, how="all")

## apd.dropna(axis=1)

#' 
#' 
#' Fill in missing data
#' ===
#' 
## apd.fillna(-99)

## #apd.fillna(apd.mean()) ## fill by mean alue

## apd.fillna({1:-1, 2:-2, 4:-4})

## apd.fillna(-99, inplace=True)

## apd

#' 
#' 
#' Fill in missing data using interpolation
#' ===
#' 
#' - ffill: interpolation using forward (next) value
#' - bfill: interpolation using backword (previous) value
#' 
## arr = pd.Series([0,np.nan,3,np.nan, 5, np.nan])

## #apd.fillna(apd.mean()) ## fill by mean alue

## arr.fillna(method="ffill")

## arr.fillna(method="bfill")

#' 
#' Filtering
#' ===
#' 
#' - np.where
#' 
## apd = pd.DataFrame({"c1": [1,2,3,4], "c2": ["a", "b", "c", "d"]})

## apd["c1"] > 2

## np.where(apd["c1"] > 2)

## apd.iloc[np.where(apd["c1"] > 2)]

## apd.iloc[np.where( (apd["c1"] > 2) & (apd["c2"] == "d"))]

#' 
#' Duplicates
#' ===
#' 
#' - duplicated(): return duplicated index
#' - drop_duplicates(): drop rows with duplicated index = True
#' 
## apd = pd.DataFrame({"c1": [1,1,2,2,3,3], "c2": ["a", "b", "a", "a", "a", "b"]})

## apd.duplicated()

## apd.drop_duplicates()

#' 
#' ---
#' 
#' - drop duplicates based on cerntain columns
#' 
## apd.drop_duplicates("c1")

#' 
#' - keep the last one (keep = "last")
#' 
## apd.duplicated(["c1"], keep="last")

## apd.drop_duplicates("c1", keep="last")

#' 
#' 
#' Map
#' ===
#' 
#' - A series has a map method: 
#'   - apply a function to each element of a Series
#' 
## population = pd.DataFrame({"City": ["Gainesville", "Orlando", "Tampa", "Pittsburgh", "Philadelphia"],

##     "Population": [140,309,387,300,1576]}

## )

## city_to_state = {"Gainesville": "FL", "Orlando": "FL", "Tampa": "FL", "Pittsburgh": "PA", "Philadelphia":"PA"}

## population

## city_to_state

## population["City"].map(lambda x: city_to_state[x])

#' 
#' ---
#' 
#' - short cut using map:
#' 
## population.City.map(city_to_state)

## population["State"] = population.City.map(city_to_state)

#' 
#' 
#' Replacing values
#' ===
#' 
## apd = pd.DataFrame([[1,2,3], [2, 3, 4], [3, 4, 5]])

## apd

## apd.replace(4, np.nan)

## apd.replace([3,4], [-3,-4])

## apd.replace({3:-3, 4:-4})

#' 
#' HW question
#' ===
#' 
## data = pd.DataFrame({"l1": [1,2,3,18,13, 1, 15,6,-99,21,3,np.nan],

## "l2": [1,np.nan,3,7,np.nan,1, 5,-99,-99,3,3,9]})

## data

#' 
#' 
#' - Perform data cleaning:
#'   - replace -99 to NA
#'   - remove missing value (rows with any missing values)
#'   - remove duplicates
#'   - sort by l1 (decending order)
#'   - convert to integer
#'   - extract the first column as Series, and using map function to calculate the sum of all digits for each element of the Series
#' 
#' 
#' 
#' 
#' 
#' 
#' Renaming Axis Indexes
#' ===
#' 
#' - the index also has the map method
#' 
## pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

## pd1

## 

## pd1.index = pd1.index.map(lambda x: x[:3])

## pd1

#' 
#' ---
#' 
#' - rename function to rename columns and indexes
#' 
## pd1.rename(index=str.upper, columns=str.upper)

## 

## pd1.rename(index={"Flo": "FL", "Tex": "TX"}, columns = {"b": "BB"})

## 

## pd1.rename(index={"Flo": "FL", "Tex": "TX"}, columns = {"b": "BB"}, inplace = True)

## 

## pd1

#' 
#' 
#' 
#' 
#' Discretization and Binning
#' ===
#' 
#' - binning a vector based on certain breaks
#' 
## ages = np.random.default_rng(32611).integers(low=0, high=100, size=10)

## bins = [0, 20, 40, 60, 80, 100]

## groups = pd.cut(ages, bins)

## groups.codes

## groups.categories

## groups.value_counts()

#' 
#' ---
#' 
#' - set labels for each categories
#' 
## groups2 = pd.cut(ages, bins, labels = ["0-20", "20-40", "40-60", "60-80", "80-100"])

## groups2.value_counts()

#' 
#' 
#' 
#' Sub-Sampling
#' ===
#' 
## rng = np.random.default_rng(32611)

## arr = pd.DataFrame(rng.standard_normal((100,3)))

#' 
#' - random sub array by rows
#' 
## index = rng.choice(range(100),3)

## arr.take(index)

## arr.sample(n=3)

#' 
#' 
#' 
#' 
#' 
#' 
#' Reference
#' ===
#' 
#' - https://wesmckinney.com/book/pandas-basics.html
#' - https://wesmckinney.com/book/accessing-data.html
#' - https://wesmckinney.com/book/time-series.html
#' - https://wesmckinney.com/book/data-cleaning.html
