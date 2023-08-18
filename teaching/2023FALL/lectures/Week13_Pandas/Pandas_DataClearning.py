#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday Nov 16th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Pandas Data Cleaning"
#' ---
#' 

#' 
#' 
import numpy as np
import pandas as pd
#' 
#' 
#' Outlines
#' ===
#' 
#' 
#' - Missing data
#' - Filtering
#' - Duplicates
#' - Map
#' - Replacement
#' - Discretization
#' - Sub-sampling
#' - String in Pandas
#' 
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
vec_withna = pd.Series([0, np.NaN, np.nan, None, 1])
vec_withna.isnull()
vec_withna.isna()
vec_withna[vec_withna.notnull()]
vec_withna[vec_withna.notna()]
vec_withna.dropna()
#' 
#' Missing data -- DataDrame
#' ===
#' 
#' - create a new variable NA = np.nan
#' - dropna: by default, drop a row if any missing value exists
#' 
from numpy import nan as NA

apd = pd.DataFrame([[1,2,3], [NA, NA, NA], [4, NA, 6]])
apd.dropna()
#' 
#' - drop if all elements are missing
#' 
apd.dropna(how="all")
#' 
#' ---
#' 
#' - drop by columns
#' 
apd[4] = NA
apd
apd.dropna(axis=1, how="all")
apd.dropna(axis=1)
#' 
#' ---
#' 
#' - drop rows with NA, 
#'   - but keeping only rows with certain number of observations
#' 
apd
apd.dropna(thresh = 2) ## at least two non-missing elements
apd.dropna(thresh = 3) ## at least three non-missing elements
#' 
#' Fill in missing data
#' ===
#' 
apd.fillna(-99)
#apd.fillna(apd.mean()) ## fill by mean alue
apd.fillna({1:-1, 2:-2, 4:-4})
apd.fillna(-99, inplace=True)
apd
#' 
#' 
#' Fill in missing data using interpolation
#' ===
#' 
#' - ffill: interpolation using forward (next) value
#' - bfill: interpolation using backword (previous) value
#' 
arr = pd.Series([0,np.nan,3,np.nan, 5, np.nan])
#apd.fillna(apd.mean()) ## fill by mean alue
arr.fillna(method="ffill")
arr.fillna(method="bfill")
#' 
#' Filtering
#' ===
#' 
#' - np.where
#' 
apd = pd.DataFrame({"c1": [1,2,3,4], "c2": ["a", "b", "c", "d"]})
apd["c1"] > 2
np.where(apd["c1"] > 2)
apd.iloc[np.where(apd["c1"] > 2)]
apd.iloc[np.where( (apd["c1"] > 2) & (apd["c2"] == "d"))]
#' 
#' Duplicates
#' ===
#' 
#' - duplicated(): return duplicated index
#' - drop_duplicates(): drop rows with duplicated index = True
#' 
apd = pd.DataFrame({"c1": [1,1,2,2,3,3], "c2": ["a", "b", "a", "a", "a", "b"]})
apd.duplicated()
apd.drop_duplicates()
#' 
#' ---
#' 
#' - drop duplicates based on cerntain columns
#' 
apd.drop_duplicates("c1")
#' 
#' - keep the last one (keep = "last")
#' 
apd.duplicated(["c1"], keep="last")
apd.drop_duplicates("c1", keep="last")
#' 
#' 
#' Map
#' ===
#' 
#' - A series has a map method: 
#'   - apply a function to each element of a Series
#' 
population = pd.DataFrame({"City": ["Gainesville", "Orlando", "Tampa", "Pittsburgh", "Philadelphia"],
    "Population": [140,309,387,300,1576]}
)
city_to_state = {"Gainesville": "FL", "Orlando": "FL", "Tampa": "FL", "Pittsburgh": "PA", "Philadelphia":"PA"}
population
city_to_state
population["City"].map(lambda x: city_to_state[x])
#' 
#' ---
#' 
#' - short cut using map:
#' 
population.City.map(city_to_state)
population["State"] = population.City.map(city_to_state)
#' 
#' 
#' Replacing values
#' ===
#' 
apd = pd.DataFrame([[1,2,3], [2, 3, 4], [3, 4, 5]])
apd
apd.replace(4, np.nan)
apd.replace([3,4], [-3,-4])
apd.replace({3:-3, 4:-4})
#' 
#' In class exercise
#' ===
#' 
data = pd.DataFrame({"l1": [1,2,3,18,13, 1, 15,6,-99,21,3,np.nan], 
"l2": [1,np.nan,3,7,np.nan,1, 5,-99,-99,3,3,9]})
data
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
#' ---
#' 
#' write the code in one line
#' 
data.replace({-99:np.nan}).dropna().drop_duplicates().sort_values(by="l1",ascending=False).astype(np.int32).l1.map(lambda x: np.sum([int(i) for i in str(x)]))
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
pd1 = pd.DataFrame(np.arange(9).reshape(-1,3), columns = list("bdc"), index = ["Florida", "Texax", "Utah"])
pd1

pd1.index = pd1.index.map(lambda x: x[:3])
pd1
#' 
#' ---
#' 
#' - rename function to rename columns and indexes
#' 
pd1.rename(index=str.upper, columns=str.upper)

pd1.rename(index={"Flo": "FL", "Tex": "TX"}, columns = {"b": "BB"})

pd1.rename(index={"Flo": "FL", "Tex": "TX"}, columns = {"b": "BB"}, inplace = True)

pd1
#' 
#' 
#' 
#' 
#' Discretization and Binning
#' ===
#' 
#' - binning a vector based on certain breaks
#' 
ages = np.random.default_rng(32611).integers(low=0, high=100, size=10)
bins = [0, 20, 40, 60, 80, 100]
groups = pd.cut(ages, bins)
groups.codes
groups.categories
groups.value_counts()
#' 
#' ---
#' 
#' - set labels for each categories
#' 
groups2 = pd.cut(ages, bins, labels = ["0-20", "20-40", "40-60", "60-80", "80-100"])
groups2.value_counts()
#' 
#' 
#' 
#' Detecting Outliers
#' ===
#' 
rng = np.random.default_rng(32611)
arr = pd.DataFrame(rng.standard_normal((100,3)))
arr.head()
arr.describe()
#' 
#' ---
#' 
#' - outliers are defined as >2 or <-2 
#' 
#' - outliers for a certain column
#' 
col0 = arr[0]
col0[np.abs(col0) > 2]
#' 
#' - obtain rows with outliers (change criteria to 3)
#' 
arr[(np.abs(arr)>3).any(axis="columns")]
#' 
#' ---
#' 
#' - do a truncation and set min/min values for the outliers
#' 
arr[(np.abs(arr)>2)] = np.sign(arr) * 2
arr.describe()
#' 
#' Sub-Sampling
#' ===
#' 
rng = np.random.default_rng(32611)
arr = pd.DataFrame(rng.standard_normal((100,3)))
#' 
#' - random sub array by rows
#' 
index = rng.choice(range(100),3)
arr.take(index)
arr.sample(n=3)
#' 
#' 
#' Create dummy variables
#' ===
#' 
#' - get_dummies to create dummy variables
#' 
data = pd.DataFrame({"keys": ["b", "b", "a", "a", "c", "c"], "values": np.arange(6)})
data
dummy = pd.get_dummies(data["keys"])
dummy
#' 
#' ---
#' 
#' - add_prefix: to add prefix in a DataFrame
#' - we will learn about the join method next week
#' 
dummy.add_prefix("Group_")
data_combine = data.join(dummy.add_prefix("Group_"))
data_combine
#' 
#' String in Series
#' ===
#' 
#' - str method for strings in Pandas
#' 
data = pd.Series({"Alex": "alex@gmail.com", "Beth": "BETH@yahoo.com", "Carl": "Carl@ufl.edu"})
data.str.contains("ufl")
pattern = "@|\."
data.str.split(pattern)
pattern = "(\w+)@(\w+)\.(\w+)"
data.str.findall(pattern)
data.str.findall(pattern).map(lambda x: x[0][1])
#' 
#' 
#' 
#' 
#' Reference
#' ===
#' 
#' - https://wesmckinney.com/book/data-cleaning.html
