#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' author: Zhiguang Huo (Caleb)
#' date: "Monday Nov 28th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' subtitle: "seaborn: statistical data visualization"
#' ---
#' 

#' 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#' 
#' Outlines
#' ===
#' 
#' - relational plot
#'   - scatter plot
#'   - line plot
#' - distributional plot  
#'   - histogram
#'   - density plot
#' - categorical data
#'   - barplot
#'   - boxplot
#'   - jitter plot
#' - others
#'   - pairplot
#'   - heatmap
#' 
#' 
#' Get started
#' ===
#' 
#' - package: seaborn
#' - load the package
#' 
import seaborn as sns
#' 
#' - Seaborn is a Python data visualization library based on matplotlib. 
#' - It provides a high-level interface for drawing attractive and informative statistical graphics.
#' 
#' 
#' 
#' Overview of the seaborn functionality
#' ===
#' 
#' ![](../figure/sns_overview.png)
#' 
#' 
#' tips dataset
#' ===
#' 
#' - dataset for scatter plot
#' 
tips = sns.load_dataset("tips")
tips.head()
#' 
#' - if you have trouble load the dataset, try the following code
#' 
#' ```
#' import ssl
#' ssl._create_default_https_context = ssl._create_unverified_context
#' ```
#' 
#' 
#' 
#' relplot (scatter plot)
#' ===
#' 
#' - scatter plot:
#'   - data: a dataframe
#'   - x: x axis
#'   - y: y axis
#' 
#' - implementation:
#'   - sns.relplot with kind: scatter (default)
#'   - sns.scatterplot
#' 
#' 
sns.relplot(data=tips,x="total_bill", y="tip", kind='scatter')
# sns.scatterplot(data=tips, x="total_bill", y="tip", )
#' 
#' 
#' color and style
#' ===
#' 
#' - hue: Grouping variable that will produce elements with different colors
#' 
sns.relplot(data=tips,x="total_bill", y="tip", hue = "smoker")
#' 
#' - style: shape of the points
#' 
sns.relplot(data=tips,x="total_bill", y="tip", hue = "smoker", style="smoker")
#' 
#' 
#' size and transparency
#' ===
#' 
#' - size: size of points
#' 
sns.relplot(data=tips,x="total_bill", y="tip", size = "size")
#' 
#' - alpha: transparency
#' 
sns.relplot(data=tips,x="total_bill", y="tip", size = "size", alpha = 0.5)
#' 
#' 
#' 
#' 
#' 
#' 
#' multiple subfigures
#' ===
#' 
#' - col: columns
#' 
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)
#' 
#' 
#' 
#' scatter plot with linear regression fitting
#' ===
#' 
sns.lmplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")
#' 
#' 
#' 
#' styles
#' ===
#' 
#' - There are 5 built-in styles
#' 	- darkgrid (default)
#' 	- whitegrid
#' 	- dark
#' 	- white
#' 	- ticks
#' 
sns.set_style("whitegrid")
sns.relplot(data=tips,x="total_bill", y="tip", kind='scatter')
#' 
#' alternative function: 
#' 
#' ```
#' sns.set_theme(style = "whitegrid")
#' ```
#' 
#' Scaling Plots -- set_context
#' ===
#' 
#' - There are 4 built-in context (related to font size)
#' 	- notebook (default)
#' 	- paper
#' 	- talk
#' 	- poster
#' 
#' 
sns.set_style("whitegrid")
sns.set_context("talk")
sns.relplot(data=tips,x="total_bill", y="tip", kind='scatter')
#' 
#' alternative function: 
#' 
#' ```
#' sns.set_theme(context = "talk, style = "whitegrid")
#' ```
#' 
#' 
#' fmri dataset
#' ===
#' 
#' - dataset for line plot
#' 
fmri = sns.load_dataset("fmri")
fmri.head()
fmri_sub13 = fmri[(fmri["subject"]=="s13")]
fmri_sub13_stim = fmri_sub13[(fmri_sub13["event"]=="stim")]
fmri_sub13_stim_parietal = fmri_sub13_stim[(fmri_sub13_stim["region"]=="parietal")]
fmri_sub13_stim_parietal.head()
#' 
#' 
#' relplot (line plot)
#' ===
#' 
#' - line plot:
#'   - data: a dataframe
#'   - x: x axis
#'   - y: y axis
#'   
#' - implementation:
#'   - sns.relplot with kind = line
#'   - sns.lineplot
#' 
#' 
sns.relplot(data=fmri_sub13_stim_parietal, kind="line", x="timepoint", y="signal")
# sns.lineplot(data=fmri_sub13_stim_parietal, x="timepoint", y="signal")
#' 
#' combine scatter and line plots
#' ===
#' 
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=fmri_sub13_stim_parietal, x="timepoint", y="signal", ax=ax)
sns.lineplot(data=fmri_sub13_stim_parietal, x="timepoint", y="signal", ax=ax)
#' 
#' relplot (line plot)
#' ===
#' 
#' - line plot:
#' 	- hue: Grouping variable that will produce elements with different colors
#' 	- style: shape
#' 	- size: size
#' 	- col: columns
#' 
#' 
#' - color by hue
#' 
sns.relplot(data=fmri_sub13_stim, kind="line", x="timepoint", y="signal", hue="region", )
#' 
#' ---
#' 
#' - style
#' 
sns.relplot(data=fmri_sub13, kind="line", x="timepoint", y="signal", hue="region", style = "event", )
#' 
#' - columns
#' 
sns.relplot(data=fmri_sub13, kind="line", x="timepoint", y="signal", hue="region", col="event", )
#' 
#' multiple subjects
#' ===
#' 
#' - If multiple data exist for the same data points, the mean value and the confidence interval will be plotted
#' 
sns.relplot(data=fmri, kind="line", x="timepoint", y="signal", hue="region", col="event", )
#' 
#' Exercise
#' ===
#' 
#' - based on the following fmri sub dataset, draw a spaghetti plot of signal with respect to time point for each subject
#' 
fmri_sub = fmri[(fmri["event"]=="stim") & (fmri["region"]=="parietal")]
fmri_sub.head()
#' 
#' ---
#' 
#' - solution
#' 
sns.relplot(data=fmri_sub, kind="line", x="timepoint", y="signal", hue="subject", )
#' 
#' 
#' histogram
#' ===
#' 
#' - sns.displot
#'   - kind="hist" by default
#'   - kind="kde": kernal density estimation
#'   - kind="ecdf": emperical cdf
#'   
# sns.displot(data=tips, x="total_bill")
sns.displot(data=tips, x="total_bill", col="time", kde=True)
## density function only 
## sns.displot(data=tips, x="total_bill", col="time", kind = "kde")
#' 
#' 
#' empirical cumulative distribution function
#' ===
#' 
sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker")
#' 
#' - rug: show each observation with marginal ticks
#' 
#' ```
#' # sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker", rug=True)
#' ```
#' 
#' 
#' Visualizing categorical data
#' ===
#' 
#' - Categorical scatterplots:
#'   - stripplot() (with kind="strip"; the default)
#'   - swarmplot() (with kind="swarm")
#' 
#' - Categorical distribution plots:
#'   - boxplot() (with kind="box")
#'   - violinplot() (with kind="violin")
#' 
#' - Categorical estimate plots:
#'   - pointplot() (with kind="point")
#'   - barplot() (with kind="bar")
#'   - countplot() (with kind="count")
#' 
#' 
#' 
#' catplot (strip plot)
#' ===
#' 
#' - strip plot: jitter plot
#' 
#' 
sns.catplot(data=tips, kind="strip", x="day", y="total_bill", hue="smoker")
#' 
#' ```
#' sns.catplot(data=tips, kind="strip", x="day", y="total_bill", hue="smoker", dodge=True)
#' sns.stripplot(data=tips, x="day", y="total_bill", hue="smoker")
#' ```
#' 
#' catplot (swarm plot)
#' ===
#' 
#' - swarm plot
#' 
#' 
sns.catplot(data=tips, kind="swarm", x="day", y="total_bill", hue="smoker", dodge=True)
#' 
#' ```
#' sns.catplot(data=tips, kind="swarm", x="day", y="total_bill", hue="smoker", )
#' sns.swarmplot(data=tips, x="day", y="total_bill", hue="smoker")
#' ```
#' 
#' 
#' catplot (boxplot)
#' ===
#' 
#' - boxplot
#' 	- sns.catplot with kind = "box"
#' 	- sns.boxplot
#' 
sns.catplot(data=tips, kind="box", x="day", y="total_bill", hue="smoker")
# sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")
#' 
#' overlay box plot and jitter plot
#' ===
#' 
fig, ax = plt.subplots(figsize=(6, 4))
sns.stripplot(data=tips, x="day", y="total_bill", hue="smoker", dodge=True, ax=ax, legend=False)
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker", ax=ax)
#' 
#' catplot (violin)
#' ===
#' 
#' - violin
#' 	- sns.catplot with kind = "violin"
#' 	- sns.violinplot
#' 
sns.catplot(data=tips, kind="violin", x="day", y="total_bill", hue="smoker")
# sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker")
#' 
#' 
#' catplot (pointplot)
#' ===
#' 
#' - pointplot: point estimate + error bar
#' 	- sns.catplot with kind = "point"
#' 	- sns.pointplot
#' 
sns.catplot(data=tips, kind="point", x="day", y="total_bill", hue="smoker")
# sns.pointplot(data=tips, x="day", y="total_bill", hue="smoker")
#' 
#' 
#' catplot (barplot)
#' ===
#' 
#' - barplot: 
#' 	- sns.catplot with kind = "bar"
#' 	- sns.barplot
#' 
sns.catplot(data=tips, kind="bar", x="day", y="total_bill", hue="smoker")
# sns.barplot(data=tips, x="day", y="total_bill", hue="smoker")
# sns.barplot(data=tips, x="day", y="total_bill", hue="smoker", ci=None)
# sns.barplot(data=tips, y="day", x="total_bill", hue="smoker") ## horizontal barplot
#' 
#' 
#' catplot (countplot)
#' ===
#' 
#' - countplot
#' 	- sns.catplot with kind = "count"
#' 	- sns.countplot
#' 
sns.catplot(data=tips, kind="count", x="day", hue="smoker")
# sns.countplot(data=tips, x="day", hue="smoker")
#' 
#' 
#' iris dataset
#' ===
#' 
#' - dataset for heatmap and pairplot
#' 
iris = sns.load_dataset("iris")
iris.head()
#' 
#' 
#' 
#' pairplot
#' ===
#' 
#' - to visualize the relationship between each pair of numerical variable
#' 
p1 = sns.pairplot(iris, hue="species")
#sns.pairplot(iris)
plt.show()
#' 
#' 
#' heatmap
#' ===
#' 
iris_sub = iris.drop(columns="species")
sns.heatmap(iris_sub)
#' 
#' 
#' Reference
#' ===
#' 
#' https://seaborn.pydata.org/tutorial
#' 
