#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Nov 14th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Pandas Basics"
#' ---
#' 

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
import numpy as np
#' 
import pandas as pd
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
aseries = pd.Series([1,2,3])
aseries
aseries.values
aseries.index ## like range(3)
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
bseries = pd.Series([1.,2.2,3.5], index=['a', 'b', 'c'])
bseries
bseries.index ## like range(3)
## [i for i in bseries.index]
## list(bseries.index)
#' 
#' 
#' Series
#' ===
#' 
#' - Series is like a numpy array
#' 
bseries*10
np.log(bseries)
#' 
#' 
#' Series
#' ===
#' 
#' - series is like a dictionary
#' 
bseries['a']
bseries.a
bseries[['a','b']]
'a' in bseries
#' 
#' ---
#' 
#' - series can be created via a dictionary
#' 
adict = {"a":1, "b":2, "c":3}
pd.Series(adict) 
#' 
#' - we can also specify indexes when creating Series
#' 
cseries = pd.Series(adict, index=["a", "c", "b"]) 
cseries
#' 
#' ---
#' 
#' - what if we specify indexes that doesn't exist
#' 
dseries = pd.Series(adict, index=["d", "c", "b"]) 
dseries ## nan will be created
pd.isnull(dseries)
#' 
#' 
#' Series meta data
#' ===
#' 
#' - meta data for Series
#' 
bseries.name="XX"
bseries.index.name="YY"
## bseries
#' 
#' - the Series index can be altered in-place by assignment
#' 
cseries.index
cseries.index = ["aa", "bb", "cc"]
cseries
#' 
#' 
#' Series indexing 
#' ===
#' 
#' - index
#' 
obj = pd.Series(np.arange(3), index=["a", "b", "c"])
obj
obj.index ## index is like range()
obj.index[:-1]
#' 
#' - index is not directly mutable
#' 
#' ```
#' obj.index[0] = "x"
#' ```
#' 
#' Reindexing 
#' ===
#' 
#' - Series
#'   - reorder the index
#'   
data = pd.Series([1.1, 2.2, 3.3], index = ['b', 'c', 'a'])
data
data2 = data.reindex(['a', 'b', 'c', 'd'])
data2
#' 
#' 
#' 
#' 
#' Drop entries in Series
#' ===
#' 
#' - by default, not in place operator (inplace = False)
#' 
data = pd.Series(np.arange(4), index = ['a', 'b', 'c', 'd'])
data
data.drop("a") ## not inplace operator
data2 = data.drop(["a", "b"])
data2
#' 
#' 
#' Series entry selection
#' ===
#' 
#' - by index
#' 
data = pd.Series([1,2,3,4], index=["a", "b", "c", "d"])
data["b"]
#' 
#' 
#' - by row ID
#' 
#' 
data[1]
data[1:3]
#' 
#' 
#' Series entry selection (more)
#' ===
#' 
data[[1,3]]
data[data > 2] ## by logical array
data["b":"c"]
##data["b":"c"] = 0
#' 
#' Series entry selection by loc/iloc
#' ===
#' 
data = pd.Series([1,2,3,4], index=["a", "b", "c", "d"])
#' 
#' - loc: by index name
#' 
data.loc["a"]
data.loc[["a", "c"]]
#' 
#' ---
#' 
#' - loc: by row ID
#' 
data.iloc[1]
data.iloc[1:]
data.iloc[-2]
#' 
#' Series membership
#' ===
#' 
data = pd.Series(list("abcdadcccdaabbb"))
#' 
#' - unique
#' 
data.unique()
#' 
#' - value counts
#' 
data.value_counts()
#' 
pd.value_counts(data, sort = False)
#' 
#' Series membership
#' ===
#' 
#' - membership
#' 
mask = data.isin(["a", "b"])
data[mask]
#' 
uniq_val = data.unique()
pd.Index(uniq_val).get_indexer(data)
#' 
#' Arithmetic and data alignment
#' ===
#' 
#' 
a1 = pd.Series([1,2,3], index = ["a", "b", "c"])
a2 = pd.Series([2,3,4], index = ["a", "b", "c"])
a3 = pd.Series([2,3,4], index = ["b", "c", "d"])
#' 
#' - if indexes are the same
#' 
a1 + a2
#' 
#' - if indexes are different
#' 
a1 + a3
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
data = {"Name": ["Amy", "Beth", "Carl"],
        "Age": [24, 22, 19],
        "Sex": ["F", "F", "M"]        
}
data
pd.DataFrame(data)
#' 
#' ---
#' 
#' - reorder the columns
#' 
pd.DataFrame(data, columns = ["Sex", "Age", "Name"])
#' 
#' - if a column name doesn't exist, missing value will be created
#' 
pd.DataFrame(data, columns = ["Sex", "Age", "Name", "Email"])
#' 
#' - if there was no user-defined index, the index argument will create indexes
#' 
apd = pd.DataFrame(data, columns = ["Sex", "Age", "Name"], index = ["a", "b", "c"])
apd
#' 
#' 
#' reset_index and set_index
#' ===
#' 
apd
data2 = apd.reset_index()
data2
data2.set_index("index")
#data2.set_index("Name")
#' 
#' 
#' DataFrame - select columns
#' ===
#' 
#' - select columns
#' 
apd["Age"]
apd.Age
apd[["Name", "Age"]]
#' 
#' DataFrame - select rows
#' ===
#' 
#' - by row index
#' 
## apd[1] # this won't work
apd[1:]
apd[apd["Age"] > 20]
#' 
#' Selection with loc
#' ===
#' 
#' - loc: by row/column name
#' 
apd.loc["a"]
apd.loc[:,"Name"]
apd.loc["a", ["Name", "Age"]]
#' 
#' 
#' Selection with iloc
#' ===
#' 
#' - iloc: by row/column integer index
#' 
apd.iloc[0]
apd.iloc[-1]
apd.iloc[:,1]
apd.iloc[0, [0,1]]
#' 
#' 
#' create columns
#' ===
#' 
#' 
apd["Age20"] = apd.Age >= 20
apd
apd["debt"] = np.nan
apd
#' 
#' 
#' modify columns
#' ===
#' 
apd["debt"] = 15.1
apd
apd["debt"] = np.arange(3.)
apd
#' 
#' - if the assigned value is a Series, only the element with existing indexes will be assigned
#' 
val = pd.Series([5., 7.7], index=["a", "c"])
apd["debt"] = val
apd
#' 
#' 
#' more on columns for pd.DataFrame
#' ===
#' 
#' - get column names
#' 
apd.columns
#' 
#' - delete columns
#' 
del apd["debt"]
apd.columns
#' 
#' - transpose the DataFrame
#' 
apd.T
#' 
#' 
#' create DataFrame via nested dictionary
#' ===
#' 
#' - when pass a nested dictionary to pd.DataFrame:
#'     - outer dict keys will be columns
#'     - inner dict keys will be row index
#' 
pop = {"Florida": {2020: 10.9, 2021: 11.3, 2022: 13.4},
        "Texas": {2020: 20.5, 2021: 21.1}
}
bpd = pd.DataFrame(pop)
bpd
#' 
#' 
#' some extra operations on pd.DataFrame
#' ===
#' 
#' 
#' - set meta data
#' 
bpd.index.name = "Year"
bpd.columns.name = "States"
bpd
#' 
#' 
#' some attributes of pd.DataFrame
#' ===
#' 
#' - values attribute: return numpy array
#' 
bpd.values
#' 
#' - index attribute
#' 
bpd.index
2021 in bpd.index
#' 
#' - columns attribute
#' 
bpd.columns
"Florida" in bpd.columns
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
data = pd.DataFrame(np.arange(6).reshape(3,-1), index =['x', 'y', 'z'], columns = ["Florida", 'Texas'] )
data
data2 = data.reindex(['x', 'y', 'z', 'w'])
data2
#' 
#' ---
#' 
#' - reindexing columns
#' 
states = ['Utah', 'Florida', 'Texas']
data3 = data2.reindex(columns = states)
data3
data3.loc[["x", "y"], states]
#' 
#' 
#' Drop entries in DataFrame
#' ===
#' 
#' - by default, drop index by rows (axis=0)
#' - to drop columns, set axis = 1 or axis = "columns"
#' - by default, not in place operator (inplace = False)
#' 
data0 = {"Name": ["Amy", "Beth", "Carl"],
        "Age": [24, 22, 19],
        "Sex": ["F", "F", "M"]        
}
data = pd.DataFrame(data0, index = ["1", "2", "3"])
data.drop("1") ## drop by index
data.drop("Name", axis=1)
data.drop(["Name", "Age"], axis="columns")

## data.drop("1", inplace=True); data
#' 
#' 
#' Data alignment for DataFrame
#' ===
#' 
#' 
pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])
pd2 = pd.DataFrame(np.arange(12).reshape(-1,3), columns = list("bac"), index = ["Florida", "Texax", "Utah", "Ohio"])

pd1
pd2
pd1 + pd2
#' 
#' 
#' 
#' Arithmetic for DataFrame
#' ===
#' 
pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])
pd3 = pd.DataFrame(np.arange(1,10).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

pd1.add(pd3)
pd1.sub(1)
pd1.rsub(1)
#' 
#' 
#' Operation between DataFrame and Series
#' ===
#' 
pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])

series1 = pd1.iloc[0]

pd1

series1

pd1 - series1
#' 
#' Sorting and Ranking
#' ===
#' 
#' - Series
#' 
series1 = pd.Series([1,3,2,4], index = ['b', 'd', 'a', 'c'])
series1
series1.sort_index()
series1.sort_index(ascending=False)
series1.sort_values()
#' 
#' 
#' Sorting and Ranking
#' ===
#' 
#' - DataFrame
#' 
pd1 = pd.DataFrame(np.array([3,2,5,1,4,6]).reshape(3,2), index = ['c', 'a', 'b'], columns = ["x", "y"])
pd1.sort_index()
pd1.sort_values(["x"])
pd1.sort_values(["y", "x"]) ## ## multiple criteria
#' 
#' 
#' A Stock Example Data
#' ===
#' 
import pandas_datareader.data as web

stock_TSLA = web.get_data_yahoo("TSLA")
# stock_TSLA
stock_TSLA.keys()
stock_data = {symbol: web.get_data_yahoo(symbol) for symbol in {"AAPL", "TSLA", "GOOG", "META"}}
price = pd.DataFrame({symbol: data['Adj Close'] for symbol, data in stock_data.items()})
price.head()
#' 
#' ---
#' 
price.head(n=3)
returns = price.pct_change()
returns.tail()
return1000 = returns[-1000:]
return1000.shape
#' 
#' 
#' 
#' Function application
#' ===
#' 
return1000.apply(np.mean)
return1000.apply(np.mean, axis = 1)
#' 
#' 
#' Function application
#' ===
#' 
#' - lambda function
#' 
f = lambda x: x.max() - x.min()
return1000.apply(f)
return1000.apply(lambda x: x.max() - x.min())
#' 
#' - user specified functions
#' 
def f2(x):
    return pd.Series([x.min(), x.max()], index = ["min", "max"])

return1000.apply(f2)
#' 
#' 
#' 
#' 
#' 
#' Summary statistics
#' ===
#' 
#' 
return1000.sum() ## default axis = 0
return1000.sum(axis = 1)
#' 
#' ---
#' 
return1000.median()
return1000.idxmax()
#' 
#' Summary statistics
#' ===
#' 
#' 
return1000.idxmin(axis = "columns")
return1000.cumsum()
return1000.var() ## default axis = 0
## return1000.std(axis="columns")
#' 
#' 
#' Summary statistics
#' ===
#' 
return1000.describe()
#' 
#' 
#' 
#' Correlation
#' ===
#' 
#' - correlation: corr()
#' 
return1000["AAPL"].corr(return1000["TSLA"])
return1000.AAPL.corr(return1000.TSLA)
return1000.corr()
#' 
#' 
#' Covariance
#' ===
#' 
#' - covariance: cov()
#' 
return1000["AAPL"].cov(return1000["TSLA"])
return1000.cov()
#' 
#' Value counts for DataFrame
#' ===
#' 
data = pd.DataFrame({'a':[1,2,3,3], 'b':[1,1,2,4], 'c':[2,2,3,3]})
data["a"].value_counts()
data.apply(pd.value_counts)
data.apply(pd.value_counts).fillna(0)
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
import io
import requests
url="https://caleb-huo.github.io/teaching/data/Python/Student_data.csv"
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))
c.head()
#' 
#' 
#' Read_csv more options
#' ===
#' 
#' - skip header
#' 
data = pd.read_csv("Student_data.csv", header=None)
data.head()
#' 
#' - skip one line
#' 
data = pd.read_csv("Student_data.csv", header=None, skiprows = 1)
data.head()
#' 
#' 
#' Read_csv more options
#' ===
#' 
#' - set column names
#' 
data = pd.read_csv("Student_data.csv", skiprows = 1, names = ["a", "b", "c", "d", 'e'])
data.head()
#' 
#' - specify index
#' 
data = pd.read_csv("Student_data.csv", index_col = "Name")
data.head()
data.sort_index().head()
#' 
#' Save results to csv or other format
#' ===
#' 
data = pd.read_csv("Student_data.csv")
data.to_csv("mydata.csv") ## index=True as default
data.to_csv("mydata.csv", index=False)
data.to_csv("mydata.txt", sep="\t")
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
from datetime import datetime
from datetime import timedelta

now = datetime.now()
now
now.year, now.month, now.day
now.hour, now.minute, now.second
#' 
#' timedelta
#' ===
#' 
datetime1 = datetime(2022,10,13,7,30,0)
datetime2 = datetime(2022,10,10,5,20,0)
delta = datetime1 - datetime2
delta
delta.days
delta.seconds

datetime1 + timedelta(12)
datetime1 + timedelta(12, 2)
#' 
#' String and Datatime Convertion
#' ===
#' 
date1 = datetime(2022,10,13)
str(date1)
date1.strftime("%Y-%m-%d")

val1 = "2021-10-11"
datetime.strptime(val1, "%Y-%m-%d")

val2 = "10/11/2021"
datetime.strptime(val2, "%m/%d/%Y")
#' 
#' 
#' 
#' String and Datatime Convertion
#' ===
#' 
#' - datetime to string
#' 
date1 = datetime(2022,10,13)
str(date1)
date1.strftime("%Y-%m-%d")
#' 
#' - string to datetime
#' 
val1 = "2021-10-11"
datetime.strptime(val1, "%Y-%m-%d")

val2 = "10/11/2021"
datetime.strptime(val2, "%m/%d/%Y")
#' 
#' - string to datetimeIndex
#' 
dateStr = ["2021-10-11 12:00:01", "2021-10-12 03:40:01"]
pd.to_datetime(dateStr)
#' 
#' Time Series Basics
#' ===
#' 
#' - Series with datetime as index
#' 
mydate = [datetime(2022,10,13), datetime(2022,10,14), datetime(2022,10,18)]
data = pd.Series(np.arange(30,33), index=mydate)
data
data.index
#' 
#' - selecting rows with datatimeIndex is flexible
#' 
data["2022/10/13"]
data["20221014"]
data["2022-10-18"]
#' 
#' 
#' Date range
#' ===
#' 
#' - date_range()
#' 
ind1 = pd.date_range("2022-01-01", "2022-08-01")
ind1

ind2 = pd.date_range(start = "2022-01-01", periods=35)
ind2


ind3 = pd.date_range(end = "2022-01-01", periods=35)
ind3

ind4 = pd.date_range(start = "2012-01-01", periods=10, freq="MS") ## MS: month start
ind4

#' 
#' Date selections
#' ===
#' 
return1000.head()
return1000.loc["2021"] ## all rows in a year
return1000.loc["2021-10"] ## all rows in a month
return1000.loc["2021-10-12"]
return1000.loc["2021-10-12":] ## starting from a date
#' 
#' 
#' Reference
#' ===
#' 
#' - https://wesmckinney.com/book/pandas-basics.html
#' - https://wesmckinney.com/book/accessing-data.html
#' - https://wesmckinney.com/book/time-series.html
