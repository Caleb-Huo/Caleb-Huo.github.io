#' ---
#' title: "Programming basics for Biostatistics 6099"
#' author: Zhiguang Huo (Caleb)
#' date: "Thursday Nov 16th, 2023"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "Matplotlib"
#' ---
#' 
## ----setup, include=FALSE-----------------------------------------------------
library(reticulate)
use_python("/Users/zhuo/anaconda3/envs/py311/bin/python")
#use_python("/usr/local/bin/python3.10")

#' 
## import pandas as pd

## import numpy as np

#' 
#' Outlines
#' ===
#' 
#' - time series plot
#' - figure, axes and subplots
#' - color, linetype, markers
#' - scatter and line plot
#' - barplot, histogram, and density plot
#' - heatmap
#' 
#' Get started
#' ===
#' 
#' - Matplotlib: is a basic plotting library for the Python, its prefix "mat" indicates that its syntax is originated from Matlab. 
#'   - package: matplotlib
#'   - sub package: pyplot
#'   - load the package:
#' 
## import matplotlib.pyplot as plt

#' 
#' time series plot
#' ===
#' 
#' A toy example
#' 
## rng = np.random.default_rng(32611)

## data = rng.standard_normal(30)

## data_cumsum = data.cumsum()

## plt.plot(data_cumsum)

#' 
#' 
#' 
#' Figure and axes
#' ===
#' 
#' - The Figure object is the outermost container for a matplotlib graphic, which may contain multiple Axes objects. 
#' - The Axes is an object containing attributes for individual plot, which can be created by
#'   - fig.add_subplot()
#' 
#' ```
#' fig = plt.figure()  
#' ax = fig.add_subplot()
#' ax.plot()
#' ```
#' 
#'   - plt.subplots()
#'   
#' ```
#' fig, ax = plt.subplots()
#' ax.plot()
#' ```
#' 
#'   - plt.subplot(): no axes
#' 
#' ```
#' plt.figure()
#' plt.subplot()
#' plt.plot()
#' ```
#' 
#' 1. use fig.add_subplot() to create Axes
#' ===
#' 
#' Single figure
#' 
#' - add_subplot: (nrow, ncol, index)
#' 
## fig = plt.figure()  # an empty figure with no Axes

## ax = fig.add_subplot(1,1,1) # add one Axes

## ax.plot(data_cumsum)

#' 
#' 
#' ---
#' 
#' Multiple subfigures
#' 
#' - add_subplot: (nrow, ncol, index)
#'   - figsize = (width, height)
#' 
## fig = plt.figure(figsize=(6, 3))

## ax1 = fig.add_subplot(1,2,1)

## ax2 = fig.add_subplot(1,2,2)

## plt.show()

#' 
#' 2. use plt.subplots() to create Axes
#' ===
#' 
#' Single figure
#' 
## fig, ax = plt.subplots(figsize=(3, 3))  # a figure with one Axes

## ax

## ax.plot(data_cumsum)

#' 
#' ---
#' 
#' Multiple subfigures
#' 
#' - use subplots() to create multiple Axes
#' 
## fig, axs = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes

## axs

## axs[0,0].plot(data_cumsum)

#' 
#' 3. use plt.subplot() to create Axes
#' ===
#' 
## plt.figure(figsize=(6, 3))

## plt.subplot(1,2,1)

## plt.plot(data_cumsum)

## plt.subplot(122)

## plt.show()

#' 
#' 
#' colors
#' ===
#' 
#' - basic color schemes:
#' 
#' ![](../figure/basicColors.png)
#' 
## plt.plot(data_cumsum, color = "g")

#' 
#' 
#' 
#' colors
#' ===
#' 
## fig = plt.figure(figsize=(8, 2))

## ax1 = fig.add_subplot(1,4,1)

## ax2 = fig.add_subplot(1,4,2)

## ax3 = fig.add_subplot(1,4,3)

## ax4 = fig.add_subplot(1,4,4)

## 

## ax1.plot(data_cumsum, color = "b")

## ax2.plot(data_cumsum, color = "g")

## ax3.plot(data_cumsum, color = "r")

## ax4.plot(data_cumsum, color = "k")

#' 
#' linestype
#' ===
#' 
#' - solid or '-'
#' - dotted or ':'
#' - dashed or '--'
#' - dashdot or '-.'
#' 		 
#' 		 
#' 		 
## plt.figure(figsize=(6, 6))

## plt.subplot(221)

## plt.plot(data_cumsum, linestyle = "-")

## plt.subplot(222)

## plt.plot(data_cumsum, linestyle = "--")

## plt.subplot(223)

## plt.plot(data_cumsum, linestyle = "-.")

## plt.subplot(224)

## plt.plot(data_cumsum, linestyle = ":")

## plt.show()

#' 
#' 
#' 
#' 
#' markers
#' ===
#' 
#' ![](../figure/markerTypes.png)
#' 
#' 
#' markers
#' ===
#' 
## fig, axes = plt.subplots(2,3,sharex=True, sharey=True)

## axes[0,0].plot(data_cumsum, marker = "o")

## axes[0,1].plot(data_cumsum, marker = "v")

## axes[0,2].plot(data_cumsum, marker = "^")

## axes[1,0].plot(data_cumsum, marker = "D")

## axes[1,1].plot(data_cumsum, marker = "X")

## axes[1,2].plot(data_cumsum, marker = "s")

## plt.show()

#' 
#' combinations of color, markers, and linestyle
#' ===
#' 
## plt.figure(figsize=(6, 3))

## plt.subplot(121)

## plt.plot(data_cumsum, linestyle = "--", color="g", marker="o")

## plt.subplot(122)

## plt.plot(data_cumsum,"go--") ## color, marker, linestyle

## plt.show()

#' 
#' 
#' 
#' ticks and labels
#' ===
#' 
#' - create subplot and assign to ax
#'   - ax.set_xticks: set ticks for x axis
#'   - ax.set_xticklabels: set ticks labels for x axis
#'   - ax.set_xlabel: set labels for x axis
#'   - ax.set_title: set titles for the subplot
#' 
#' also works for y axis (set_yticks, set_xticklabels, etc)
#' 
#' 
## fig = plt.figure(figsize=(4, 4))

## ax = fig.add_subplot(1,1,1)

## ax.plot(data_cumsum)

## ticks = ax.set_xticks([0,10,20,30])

## labels = ax.set_xticklabels(["A", "B", "C", "D"], rotation=45, fontsize = "small")

## ax.set_xlabel("Letters")

## ax.set_title("My python plot")

#' 
#' 
#' legend
#' ===
#' 
#' - add legend by label argument
#' - create legend by  
#'   - ax.legend()
#'   - plt.legend()
#' 
#' 
## fig = plt.figure(figsize=(4, 4))

## ax = fig.add_subplot(1,1,1)

## ax.plot(rng.standard_normal(30).cumsum(), "r--", label="red")

## ax.plot(rng.standard_normal(30).cumsum(), "bD-", label="blue")

## ax.plot(rng.standard_normal(30).cumsum(), "go-.", label="green")

## ax.legend(loc="best")

## plt.show()

#' 
#' Summary of plt attributes
#' ===
#' 
#' ![](../figure/anatomy.png){width=60%}
#' 
#' 
#' Scatter plot
#' ===
#' 
#' - prepare data as individual vectors
#'   - np.series
#'   - list
#' 
## rng = np.random.default_rng()

## data1 = rng.standard_normal(30)

## data2 = rng.standard_normal(30)

## 

## plt.figure(figsize=(4, 4))

## plt.subplot()

## plt.scatter(data1, data2)

#' 
#' Scatter plot
#' ===
#' 
#' - prepare data as pd.DataFrame
#' 
## fig = plt.figure(figsize=(4, 4))

## ax = fig.add_subplot(1,1,1)

## 

## rng = np.random.default_rng()

## data1 = rng.standard_normal(30)

## data2 = rng.standard_normal(30)

## df = pd.DataFrame({"A": data1, "B": data2})

## ax.scatter("A", "B", data=df)

## plt.show()

#' 
#' 
#' Scatter plot
#' ===
#' 
#' - use pd.DataFrame .plot methods
#' 
## rng = np.random.default_rng()

## data1 = rng.standard_normal(30)

## data2 = rng.standard_normal(30)

## df = pd.DataFrame({"A": data1, "B": data2})

## df.plot.scatter("A", "B")

#' 
#' 
#' ---
#' 
#' - use pd.DataFrame .plot methods with axes
#'   - specify axes in the ax argument
#' 
## fig, axes = plt.subplots(1,1,figsize=(3, 3))

## rng = np.random.default_rng()

## data1 = rng.standard_normal(30)

## data2 = rng.standard_normal(30)

## df = pd.DataFrame({"A": data1, "B": data2})

## df.plot.scatter("A", "B", ax=axes)

#' 
#' line plot
#' ===
#' 
#' - prepare data as individual vectors
#'   - np.series
#'   - list
#' 
## a = np.array([1,2,3,6,8,9])

## b = np.random.default_rng(32608).random(6)

## plt.plot(a,b)

#' 
#' 
#' line plot
#' ===
#' 
#' - prepare data as pd.DataFrame
#' 
## a = np.array([1,2,3,6,8,9])

## b = np.random.default_rng(32608).random(6)

## df = pd.DataFrame({"A":a, "B":b})

## plt.plot("A","B", data=df)

#' 
#' 
#' 
#' line plot
#' ===
#' 
#' - use pd.DataFrame .plot methods
#' 
## a = np.array([1,2,3,6,8,9])

## b = np.random.default_rng(32608).random(6)

## df = pd.DataFrame({"A":a, "B":b})

## df.plot.line("A","B")

#' 
#' 
#' 
#' ---
#' 
#' - use pd.DataFrame .plot methods with axes
#'   - specify axes in the ax argument
#' 
## fig, axes = plt.subplots(1,1,figsize=(3, 3))

## a = np.array([1,2,3,6,8,9])

## b = np.random.default_rng(32608).random(6)

## df = pd.DataFrame({"A":a, "B":b})

## df.plot.line("A","B", ax=axes)

#' 
#' Categorical variable
#' ===
#' 
#' - bar plot
#' 
## df = pd.DataFrame({"names":['A', 'B', 'C'], "values":[1,2,3]})

## plt.figure(figsize=(9, 3))

## plt.subplot(131)

## plt.bar("names", "values", data=df)

## plt.subplot(132)

## plt.scatter("names", "values", data=df)

## plt.subplot(133)

## plt.plot("names", "values", data=df)

## plt.suptitle('Categorical Plotting')

## plt.show()

#' 
#' 
#' Bar plot
#' ===
#' 
#' - use DataFrame plot methods
#' - horizontal bar plot
#' 
#' 
## df = pd.DataFrame({"names":['A', 'B', 'C'], "values":[1,2,3]})

## fig, axes = plt.subplots(1,2,figsize=(6, 3))

## df.plot.bar("names", "values", ax=axes[0], color="b")

## df.plot.barh("names", "values", ax=axes[1], color="g")

#' 
#' 
#' Bar plot with subgroups
#' ===
#' 
## pd1 = pd.DataFrame(np.random.rand(4,3), index = list("abcd"), columns = ["Florida", "Texax", "Utah"])

## pd1.columns.name="Columns"

## pd1.index.name="States"

## 

## fig = plt.figure(figsize=(6, 3))

## ax = fig.add_subplot(1,2,1)

## pd1.plot.bar(ax = ax)

## 

## bx = fig.add_subplot(1,2,2)

## pd1.plot.barh(ax = bx, stacked=True)

## plt.show()

#' 
#' 
#' 
#' Histogram and density plot
#' ===
#' 
## rng = np.random.default_rng(32611)

## data = rng.standard_normal(100)

## plt.hist(data, bins=20)

## plt.show()

#' 
#' 
#' Histogram and density plot
#' ===
#' 
## data2 = pd.DataFrame({"data": data})

## fig, axes = plt.subplots(1,3,figsize=(9, 3))

## data2.plot.hist(bins=20, ax = axes[0])

## data2.plot.density(ax = axes[1])

## data2.plot.hist(bins=20,density=True, ax = axes[2])

## data2.plot.density(ax = axes[2])

## plt.show()

#' 
#' 
#' heatmap
#' ===
#' 
#' - plt.axes: Add an Axes to the current figure and make it the current Axes.
#' 	- argument: a 4-tuple of floats rect = [left, bottom, width, height]
#' 
## np.random.seed(32608)

## plt.subplot(211)

## plt.imshow(np.random.random((100, 100)))

## plt.subplot(212)

## plt.imshow(np.random.random((100, 100)))

## cax = plt.axes([0.85, 0.1, 0.075, 0.8])

## plt.colorbar(cax=cax)

## plt.show()

#' 
#' 
#' Reference
#' ===
#' 
#' - https://wesmckinney.com/book/plotting-and-visualization.html
#' - https://matplotlib.org/stable/tutorials/introductory
#' 
