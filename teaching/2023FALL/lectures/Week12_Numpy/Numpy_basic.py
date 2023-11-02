#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Tuesday Nov 2nd, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Numpy basics"
#' ---
#' 
## ----setup, include=FALSE---------------------------------------
library(reticulate)
#use_python("/usr/local/bin/python3.11")
use_python("/Users/zhuo/anaconda3/envs/py311/bin/python")

#' 
#' 
#' 
#' Outlines
#' ===
#' 
#' - numpy get started
#' - numpy ndarray: multi-dimensional array object
#' - random number generator
#' - indexing and slicing
#' - universal functions: fast element-wise array functions
#' - statistical functions
#' - other array operators
#' - matrix operations
#' 
#' 
#' Numpy
#' ===
#' 
#' - Numpy: short for Numerical Python, is one of the most fundamental packages for numerical computing in Python
#'     - fast vectorized array operations
#'     - efficient descriptive statistics and aggregating data
#'     
#' - usually, import numpy as np 
#' 
## import numpy as np

#' 
#' - install numpy:
#' 
#' ```
#' https://numpy.org/install/
#' ```
#' 
#' get started
#' ===
#' 
#' - to compare the efficiency
#'     - numpy operation
#'     - list + loop
#' 
#' ```
#' n = 1000000
#' my_arr = np.arange(n)
#' my_list = list(range(n))
#' ```
#' 
#' ```
#' ## run %timeit in jupyter notebook
#' 
#' %timeit my_arr2 = my_arr * 2
#' 
#' %timeit my_lis2 = [i * 2 for i in my_list]
#' ```
#' 
#' Create ndarrays with np.array() method
#' ===
#' 
#' - ndarray: N-dimensional array object
#' 
#' - 1d array
#' 
## data1 = [1.1, 2.2, 3.3, 4.4]

## data1d = np.array(data1)

## data1d

#' 
#' - 2d array
#' 
## data2 = [[1,2,3,4], [5,6,7,8]]

## data2d = np.array(data2)

## data2d

#' 
#' 
#' 
#' Basic attributes of ndarray
#' ===
#' 
#' - number of dimension
#' 
## data2d

#' 
#' - number of elements along each dimension
#' 
## data2d.shape

#' 
#' - total number of elements along all dimensions
#' 
## data2d.size

#' 
#' - data type for any element of the ndarray
#' 
## data2d.dtype

#' 
#' 
#' 
#' 
#' numpy ndarray
#' ===
#' 
#' - ndarray for vectorized calculation.
#' 
#' - vectorized (element-wise) calculation with a scalar
#' 
## data1d * 10

#' 
#' 
#' - vectorized (element-wise) calculation with an ndarray
#' 
## data2d + data2d

#' 
#' 
#' 
#' Create ndarrays with zeros, ones, full.
#' ===
#' 
#' - zeros
#' 
## np.zeros(10)

## np.zeros((2,3))

#' 
#' - ones
#' 
## np.ones(10)

## np.ones((2,3))

#' 
#' ---
#' 
#' - full with a default value
#' 
## np.full(10, 5)

## np.full((2,3), 5)

#' 
#' ---
#' 
#' - based on existing shape
#' 
## data2 = np.array([[1,2,3,4], [5,6,7,8]])

## np.zeros_like(data2)

## np.ones_like(data2)

## np.full_like(data2, 5)

#' 
#' 
#' Create ndarrays with eye and identity
#' ===
#' 
#' - create an identity matrix
#' 
## np.eye(3)

## 

## np.identity(3)

#' 
#' 
#' Create ndarrays with Python range function
#' ===
#' 
## np.arange(10)

## 

## np.arange(3,10)

## 

## np.arange(3,10, 2)

## 

## np.arange(10,3, -1)

#' 
#' 
#' data types for ndarrays
#' ===
#' 
#' - float: np.float64
#' - int: np.int32
#' - string: np.string_
#' - Boolean: np.bool_
#' - many more...
#' 
## arr1 = np.array([1,2,3], dtype=np.float64)

## arr2 = np.array([1,2,3], dtype=np.int32)

## arr3 = np.array([1.1,2.3,3], dtype=np.string_)

#' 
#' 
## arr1.dtype; arr2.dtype; arr3.dtype;

## arr1.astype(np.int32)

## arr3.astype(np.float64)

#' 
#' data types conversions for ndarrays
#' ===
#' 
## arr = np.arange(10, dtype=np.int32)

## arr

## 

## 

## calibers = np.array([0.22, 0.54, 0.55, 0.0], dtype=np.float64)

## 

## arr.astype(calibers.dtype)

## 

## arr.astype(np.bool_)

## 

#' 
#' Arithmetic with NumPy Arrays
#' ===
#' 
#' - element wise calculation
#' 
## arr = np.array([[1., 2., 3.], [4., 5., 6.]])

## arr

## arr * arr

## arr - arr

#' 
#' 
#' 
#' Arithmetic with NumPy Arrays
#' ===
#' 
## 1 / arr

## arr ** 0.5

## arr2 = np.array([[0., 2., 11.], [3., 12., 2.]])

## arr2

## arr2 > arr

#' 
#' 
#' 
#' Reshape
#' ===
#' 
#' 
#' - basic about reshape
#' 
## arr = np.arange(8)

## arr.reshape((2,4))

## arr.reshape(2,4)

## arr.reshape(4,2)

#' 
#' ---
#' 
#' - reshape on a multidimensional array
#' 
## arr.reshape(4,2).reshape(2,4)

#' 
#' - reshape with -1:
#'     - the value used for that dimension is determined by the data
#' 
## arr.reshape(4,-1)

## arr.reshape(2,-1)

#' 
#' 
#' Pseudorandom Number Generation
#' ===
#' 
#' - random number generator:
#'   - np.random.default_rng
#'   - many other generators....
#' 
## rng = np.random.default_rng()

## print(rng)

#' 
#' 
#' generate random numbers with rng
#' ===
#' 
#' - random number between 0 and 1
#' 
## rfloat = rng.random() ## also specify size inside

## rfloat

## type(rfloat)

#' 
#' - random integers
#' 
## rints = rng.integers(low=0, high=10, size=3)

## rints

## type(rints)

#' 
#' set random seed
#' ===
#' 
## rng = np.random.default_rng(seed=42)

## arr1 = rng.random((3, 3))

## arr1

## 

## 

## rng = np.random.default_rng(seed=42)

## arr2 = rng.random((3, 3))

## arr2

#' 
#' 
#' Other random data
#' ===
#' 
#' 
#' - choice
#' 
## arr = np.arange(6)

## rng.choice(arr, size=5)

## rng.choice(arr, size=5, replace = False) ## by default, repalce is True

#' 
## arr = np.array([list(range(1,3)),list(range(3,5)),list(range(5,7))])

## rng.choice(arr, size=2)

#' 
#' 
#' Permutations (1d)
#' ===
#' 
#' - permutation: 
#'     - Randomly permute a sequence, and return the result
#' 
## rng = np.random.default_rng(seed=42)

## arr1 = np.arange(6)

## arr1

## rng.permutation(arr1)

## arr1 ## the original array remains the same

#' 
#' 
#' - shuffle: 
#'     - in place operation, modify the original array
#' 
## rng = np.random.default_rng(seed=42)

## arr1 = np.arange(6)

## rng.shuffle(arr1)

## arr1

#' 
#' 
#' Permutations (2d) -- permutation() method
#' ===
#' 
#' - permutation: 
#'     - Randomly permute a sequence, and return the result
#'     - axis = 0: permute rows
#'     - axis = 1: permute columns (within each row)
#' 
## rng = np.random.default_rng(seed=42)

## arr1 = np.arange(12).reshape(4,3)

## rng.permutation(arr1) ## default axis=0

## rng.permutation(arr1, axis=1)

#' 
#' 
#' Permutations (2d) -- shuffle() method
#' ===
#' 
#' - shuffle: 
#'     - **in place** operation, modify the original array
#'     - axis = 0: permute rows
#'     - axis = 1: permute columns (within each row)
#' 
## rng = np.random.default_rng(seed=5442)

## arr1 = np.arange(12).reshape(4,3)

## rng.shuffle(arr1) ## default axis=0

## arr1

## 

## arr1 = np.arange(12).reshape(4,3)

## rng.shuffle(arr1, axis=1)

## arr1

#' 
#' 
#' Permutations (2d) -- permutated() method
#' ===
#' 
#' - permutated: 
#'     - axis = None (default): , randomly shuffle the entire data matrix
#'     - axis = 0: permute rows fixing colunms
#'     - axis = 1: permute columns fixing rows
#' 
## rng = np.random.default_rng(seed=42)

## arr1 = np.arange(12).reshape(4,3); arr1

## # rng.permuted(arr1)

## # rng.permuted(arr1, axis=0)

## # rng.permuted(arr1, axis=1)

#' 
#' Distributions
#' ===
#' 
#' 
#' - standard normal distribution
#' 
## rng = np.random.default_rng()

## rng.standard_normal(size=10)

## rng.standard_normal(size=(2,3))

#' 
#' - general normal distribution
#' 
## rng.normal(loc=4, scale=1, size=4)

#' 
#' - more distributions:
#'   - see https://numpy.org/doc/stable/reference/random/generator.html
#' 
#' Basic indexing and slicing -- 1darray
#' ===
#' 
## arr = np.arange(10)

## arr

## 

## arr[4]

## arr[4:7]

## 

## arr[4:7] = -1

## arr

#' 
#' ---
#' 
## arr_slice = arr[4:7]

## arr_slice

## 

## arr_slice[1] = 99

## arr

## 

## arr_slice[:] = 88

## arr

#' 
#' - When we change values in arr_slice, the change is also reflected in the original array arr:
#'   - only copy the reference
#'   - advantage in memory and performance.
#' 
#' 
#' 
#' 
#' 
#' 
#' ---
#' 
#' - to make a real data copy instead of reference copy:
#'   - copy() method
#' 
## arr = np.arange(10)

## 

## arr_slice2 = arr[4:7].copy()

## arr_slice2

## 

## arr_slice2[1] = 99

## arr

## 

## arr_slice2[:] = 88

## arr

#' 
#' 
#' 
#' Basic indexing and slicing -- 2darray
#' ===
#' 
## arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])

## arr2d

## 

## arr2d[1]

## 

## arr2d[1][2]

## 

## arr2d[1,2]

#' 
#' 
#' indexing with slices -- 2darray
#' ===
#' 
## arr2d = np.array([[1,2,3], [4,5,6], [7,8,9]])

## arr2d[:2]

## arr2d[:2, 1:]

## arr2d[1, 1:]

## ## arr2d[, :1] ## this doesn't work, we have to use : to be placeholder

## arr2d[:, :1]

#' 
#' ---
#' 
## arr2d[::-1, ::-1]

## 

## arr2d[:2,1:] = 0

## arr2d

#' 
#' 
#' Boolean indexing
#' ===
## nane_list = ["Bob", "Joe", "Will", "Bob", "Will", "James"]

## names = np.array(nane_list)

## data = rng.standard_normal(size=(6,3))

## names

## data

## names == "Bob"

#' 
#' ---
#' 
## data[names == "Bob"]

## data[names == "Bob", 1:]

## names != "Bob"

## ~(names == "Bob")

## cond = names == "Bob"

## data[~cond]

#' 
#' 
#' ---
#' 
## mask = (names == "Bob") | (names == "Will")

## mask

## data[mask]

## mask2 = (names == "Bob") & (names == "Will")

## mask2

## data[mask2]

#' 
#' ---
#' 
## data[data<0] = 0

## data

## 

## data[names != "Will"] = 99

## data

#' 
#' 
#' Fancy indexing
#' ===
#' 
#' 
#' - fancy indexing is indexing using integer arrays
#' 
## data = np.arange(20).reshape(4,5)

## data[[2,0]]

## data[[-2,-4]]

#' 
#' - select elements (2,3) and (0,1)
#' 
## data[[2,0], [3,1]]

#' 
#' - select block with rows: [2,0], columns [3,1]
#' 
## data[[2,0]][:,[3,1]]

#' 
#' 
#' Transposing Arrays
#' ===
#' 
#' 
#' - transpose: T
#' 
## arr = np.arange(8).reshape([4,2])

## arr

## arr.T

#' 
#' - matrix multiplicity
#' 
## np.dot(arr.T, arr)

## arr.T @ arr

## arr.T.dot(arr)

#' 
#' Swapping Axes
#' ===
#' 
#' - Swapping axes
#' 
## arr = np.arange(16).reshape([2,2,4])

## arr

## arr.swapaxes(1,2)

#' 
#' - .T() method is a special case of swapaxes when the data is in two dimension
#' 
#' 
#' Universal Functions (ufuncs): Unary ufuncs
#' ===
#' 
#' - Universal Functions (ufuncs)
#'   - Fast Element-Wise Array Functions
#' 
#' ```
#' #arr = rng.standard_normal(6) ## try this one later
#' arr = np.arange(6)
#' arr
#' np.sqrt(arr)
#' np.exp(arr)
#' np.floor(arr); np.ceil(arr); 
#' np.log(arr), np.log10(arr); np.log2(arr); np.log1p(arr)
#' np.isnan(np.sqrt(arr))
#' np.sin(arr); np.cos(arr); np.tan(arr)
#' ```
#' 
#' ```
#' arr2 = np.array([np.nan, np.NaN, np.inf, np.Inf, 1.0])
#' arr2
#' np.isnan(arr2); np.isinf(arr2); np.isfinite(arr2)
#' ```
#' 
#' 
#' Universal Functions (ufuncs): Binary universal functions
#' ===
#' 
#' ```
#' x = rng.standard_normal(4)
#' y = rng.standard_normal(4)
#' np.maximum(x,y); np.fmax(x,y) ## fmax ignores NaN
#' np.minimum(x,y); np.fmin(x,y) ## fmax ignores NaN
#' np.add(x,y); x + y
#' np.subtract(x,y); x - y
#' np.multiply(x,y); x * y
#' np.divide(x,y); x / y
#' np.power(x,2)
#' arr_power = np.arange(len(x))
#' np.power(x,arr_power)
#' np.greater(x,y); x > y
#' np.less(x,y); x < y
#' np.greater_equal(x,y); x >= y
#' np.less_equal(x,y); x <= y
#' np.equal(x,y); x == y
#' np.not_equal(x,y); x != y
#' ```
#' 
#' Binary universal functions -- logical
#' ===
#' 
#' ```
#' x = np.array([True, True, False, False], dtype=np.bool_)
#' y = np.array([True, False, True, False], dtype=np.bool_)
#' np.logical_and(x,y); x & y
#' np.logical_or(x,y); x | y
#' np.logical_xor(x,y); x ^ y
#' ```
#' 
#' 
#' meshgrid
#' ===
#' 
## points = np.arange(-5,5,0.01) ## 1000 equally spaced points

## xs, ys = np.meshgrid(points, points)

## # xs; ys

## z = np.sqrt(xs**2 + ys**2)

## 

## import matplotlib.pyplot as plt

## plt.imshow(z)

#' 
#' Expressing conditional logic as Array Operations
#' ===
#' 
## xarr = np.array([1.1,1.2,1.4,1.5])

## yarr = np.array([2.1,2.2,2.4,2.5])

## condition = np.array([True, False, True, False])

## 

## res = [(x if c else y) for x, y, c in zip(xarr, yarr, condition)]

## res

## 

## np.array(res)

## 

## np.where(condition, xarr, yarr)

#' 
#' Expressing conditional logic as Array Operations
#' ===
#' 
## rng = np.random.default_rng(32608)

## arr = rng.standard_normal(size = (3,3))

## arr

## arr > 0

## np.where(arr>0,1,-1)

## np.where(arr>0,0,arr)

#' 
#' 
#' Statistical Methods (1darray)
#' ===
#' 
#' 
## rng = np.random.default_rng(32608)

## arr = rng.standard_normal(size = 6)

## arr

## # arr.mean(); np.mean(arr)

## # arr.sum(); np.sum(arr)

## # arr.std(); np.std(arr)

## # arr.var(); np.var(arr)

## # arr.min(); np.min(arr)

## # arr.max(); np.max(arr)

## # arr.argmin(); np.argmin(arr)

## # arr.argmax(); np.argmax(arr)

## # arr.cumsum(); np.cumsum(arr)

## # arr.cumprod(); np.cumprod(arr)

#' 
#' 
#' Statistical Methods (2darray)
#' ===
#' 
#' - np.sum()
#'   - axis=None (The default), will sum all of the elements of the input array.
#'   - axis = 0: sum over rows
#'   - axis = 1: sum over columns
#' 
#' ![](../figure/numpyAxis.jpg){width=70%}
#' 
#' - this rule also works for other numpy statistical functions.
#' 
#' 
#' Statistical Methods (2darray)
#' ===
#' 
#' 
## rng = np.random.default_rng(32608)

## arr = rng.standard_normal(size = (4,3))

## arr

## arr.sum()

## arr.sum(axis=0)

## arr.sum(axis=1)

#' 
#' 
#' Boolean arrays
#' ===
#' 
#' 
## rng = np.random.default_rng(32608)

## arr = rng.standard_normal(100)

## (arr > 0).sum()

## 

## bools = np.array([0,0,1,0], dtype=np.bool_)

## bools

## bools.any()

## bools.all()

#' 
#' sorting (1darray)
#' ===
#' 
#' - argsort(): return the index such that the array is sorted
#' 
## rng = np.random.default_rng(32608)

## arr = rng.standard_normal(5)

## arr

## arr.argsort()

## arr[arr.argsort()]

#' 
#' - sort(): in place method
#' 
#' 
## arr.sort()

## arr

#' 
#' sorting (2darray)
#' ===
#' 
#' - axis=0: sort rows
#' - axis=1 (default): sort columns within each row
#' 
## arr = np.random.default_rng(32608).standard_normal((4,3))

## arr

## arr.sort(); arr

## 

## arr = np.random.default_rng(32608).standard_normal((4,3))

## arr.sort(axis=0); arr

#' 
#' 
#' Unique and Other set logic
#' ===
#' 
#' 
#' 
## x = np.array([0,1,0,2,5,6,5,0])

## np.unique(x)

## 

## y = np.array([0,1,2,3])

## np.intersect1d(x,y)

## np.union1d(x,y)

## np.in1d(x,y)

## np.setdiff1d(x,y)

#' 
#' 
#' matrix operators
#' ===
#' 
#' - matrix multiplication
#' 
## x = np.arange(6).reshape(3,2)

## y = np.arange(6).reshape(2,3)

## 

## y.dot(x) ## np.dot(y,x)

## z = y.dot(y.T)

#' 
#' - trace, determinant
#' 
## np.trace(z)

## np.linalg.det(z)

#' 
#' - norm: $\|A\|_2 = \sqrt{\sum_{ij}a_{ij}^2}$
#' 
## np.linalg.norm(z)

#' 
#' matrix operators
#' ===
#' 
#' 
#' - matrix inverse
#' 
#' 
## np.linalg.inv(z)

## z.dot(np.linalg.inv(z))

#' 
#' - eigen value decomposition
#' 
## eigen_value, eigen_vector = np.linalg.eigh(z)

## eigen_vector.dot(np.diag(eigen_value)).dot(eigen_vector.T)

#' 
#' Reference
#' ===
#' 
#' - https://wesmckinney.com/book/numpy-basics.html
#' - https://jakevdp.github.io/PythonDataScienceHandbook/02.00-introduction-to-numpy.html
#' 
