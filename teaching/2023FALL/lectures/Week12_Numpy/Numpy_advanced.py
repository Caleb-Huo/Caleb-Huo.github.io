#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Wednesday Nov 9th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Numpy Advanced"
#' ---
#' 

#' 
#' 
#' 
#' Outlines
#' ===
#' 
#' - reshape and flatten
#' - concatenating and splitting
#' - repeat and tile
#' - take and put
#' - broadcasting
#' - aggregation
#' - file save and load
#' 
#' 
#' reshape and flatten
#' ===
#' 
import numpy as np

x = np.zeros(8).reshape(2,4)
x
x.reshape(4,-1)
x.flatten() 
x.ravel()
#' 
#' Concatenating
#' ===
#' 
#' - concatenate(): combine two np.array together
#' 
x = np.zeros(6).reshape(2,3); x
y = np.ones(6).reshape(2,3); y
np.concatenate([x,y],axis=0) ## axis=0 is default
#' 
#' ---
#' 
#' ![](../figure/numpyAxis.jpg){width=70%}
#' 
np.concatenate([x,y],axis=1)
#' 
#' Concatenating
#' ===
#' 
#' - vstack and hstack
#' 
np.vstack([x,y]) 
np.hstack([x,y]) 
#' 
#' 
#' Concatenating
#' ===
#' 
#' - c_ and r_: [] instead of ()
#' 
np.c_[x,y]
np.r_[x,y]
#' 
#' Splitting 
#' ===
#' 
rng = np.random.default_rng(32608)
data = rng.standard_normal((5,2))
np.split(data, [2]) ## default: axis = 0
x, y = np.split(data, [2])
x
y
#' 
#' 
#' Splitting 
#' ===
#' 
x, y, z = np.split(data, [1,3])
x;y;z
a, b = np.split(data, [1],axis=1)
a;b 
#' 
#' 
#' Splitting 
#' ===
#' 
x, y, z = np.vsplit(data, [1,3])
x;y;z
a, b = np.hsplit(data, [1])
a;b 
#' 
#' 
#' repeat (1darray)
#' ===
#' 
#' - repeat replicates each element of np.array some number of times
#'   - if pass in an integer, each element will be repeated that number of times
#'   - if pass in an array, each element will be repeated the number of times specified in the array
#' 
#' 
data = np.array([1,2,3])
data.repeat(2)
data.repeat([2,3,4])
#' 
#' 
#' repeat (2darray)
#' ===
#' 
#' 
data = np.arange(1,5).reshape(2,2)
data.repeat(2) ## default, flatten the array
data.repeat(2, axis=0)
data.repeat(2, axis=1)
data.repeat([2,3], axis=1)
#' 
#' 
#' tile
#' ===
#' 
#' - tile is to stack copies of an array along an axis
#'   - laying down tiles
#'   
data = np.arange(1,5).reshape(2,2)
np.tile(data, 2) 
np.tile(data, (2,1)) ## 2 rows and 1 column -- tiles
np.tile(data, (3,2)) 
#' 
#' 
#' Take and put (alternative to fancy indexing)
#' ===
#' 
#' 
#' - take: get elements
#' 
data = np.arange(0,100,10); data
inds = [9,5,2,7]
data[inds]
data.take(inds)
#' 
#' Take and put (alternative to fancy indexing)
#' ===
#' 
#' - put: assignment elements
#' 
data = np.arange(0,100,10); data
inds = [9,5,2,7]
data.put(inds, -99)
data
data.put(inds, [0,1,2,3])
data
#' 
#' 
#' Take and put (2darray)
#' ===
#' 
#' - for 2d np.array
#' 
data = np.arange(12).reshape(3,-1)
ind = [1,1,0,0]
data.take(ind, axis=None) ## default
data.take(ind, axis=0)
data.take(ind, axis=1)
#' 
#' 
#' Broadcasting
#' ===
#' 
#' Broadcasting is about arithmetic work between arrays with different shapes.
#' 
#' - 1d array and scalar
#'   - broadcast the scalar to the array (with the same shape)
#' 
data = np.arange(6)
data * 5
#' 
#' 
#' - manually convert the scalar to the array
np.full(6,5)
data * np.full(6,5)
#' 
#' Broadcasting
#' ===
#' 
#' - broadcast the 1d array to the 2d array 
#'   - the trailing dimension (e.g., 2) of the 2d array (3,2) match the length of the 1d array (e.g., shape=(2,))
#' 
data2d = np.arange(6).reshape(3,2)
vec1d = np.arange(2)
data2d + vec1d
#' 
#' - same as this (broadcasting the 1d array to the 2d array with the same shape)
#' 
vec2d = np.tile(vec1d,(3,1))
vec2d
data2d + vec2d
#' 
#' 
#' Broadcasting
#' ===
#' 
#' - broadcast the 1d array to the 2d array 
#'   - the staring dimension (e.g., 3) of the 2d array match the length of the 1d array (e.g., shape=(3,1))
#'   - the 1d array should be shaped as of dimension (n, **1**)
#' 
data2d = np.arange(6).reshape(3,2)
vec1d = np.arange(3).reshape(3,1)
data2d + vec1d
#' 
#' - same as this (broadcasting the 1d array to the 2d array with the same shape)
#' 
vec2d = np.tile(vec1d,(1,2))
vec2d
data2d + vec2d
#' 
#' Broadcasting
#' ===
#' 
#' - This will not work
#' 
#' ```
#' data2d = np.arange(6).reshape(3,2)
#' vec1d = np.arange(3)
#' data2d + vec1d
#' ```
#' 
#' 
#' Broadcasting (exercise)
#' ===
#' 
#' - the following array is given
#' 
data2d = np.arange(12).reshape(4,3)
data2d
#' 
#' - calculate the row means (mean value for each row)
#'   - subtract the row means from data2d (from each column of data2d)
#' 
#' - calculate the column means (mean value for each column)
#'   - subtract the column means from data2d (from each row of data2d)
#' 
#' 
#' Broadcasting (solution to the exercise)
#' ===
#' 
#' - subtract the row means from data2d
#'   
data2d - data2d.mean(axis=1).reshape(4,1)
#' 
#' - subtract the column means from data2d
#' 
data2d - data2d.mean(axis=0)
#' 
#' create a new dimension
#' ===
#' 
#' - use reshape
#' - use np.newaxis
#' 
data = np.ones((3,3))
data_3d = data.reshape(3,1,3)
data_3d = data[:,np.newaxis,:]
data_3d.shape
#' 
#' 3d array
#' ===
#' 
#' 
rng = np.random.default_rng(32608)
data3d = rng.standard_normal((2,3,4))
data3d.shape
depth_means = data3d.mean(axis=2) ## width, height, depth
depth_means

demeaned = data3d - depth_means[:,:,np.newaxis]
demeaned.mean(2)
#' 
#' 
#' data aggregation
#' ===
#' 
#' - reduce and accumulate methods: recuisively apply a binary ufuncs to the array
#'   - f(1,2); f(f(1,2),3), etc
#'   
data = np.arange(1,5)
np.add(data[0], data[1])
np.add.reduce(data)
np.add.accumulate(data)
np.cumsum(data)
#' 
np.multiply.reduce(data)
np.multiply.accumulate(data)
np.cumprod(data)
#' 
#' 
#' numpy file ave and load
#' ===
#' 
#' - np.save will save the numpy data in binary format, which is efficient
#' 
#' - save single numpy object 
#'   - extension: npy
#' 
#' ```
#' data = np.arange(1,10)
#' np.save("my_data.npy", data)
#' np.load("my_data.npy")
#' ```
#' 
#' - save multiple numpy objects 
#'   - extension: npz
#' 
#' ```
#' data1 = np.arange(1,10); data2 = np.arange(2,11); 
#' np.savez("my_data2.npz", x=data1, y=data2)
#' dat = np.load("my_data2.npz")
#' dat["x"]
#' dat["y"]
#' ```
#' 
#' 
#' 
#' Reference
#' ===
#' 
#' 
#' - https://wesmckinney.com/book/advanced-numpy.html
#' 
