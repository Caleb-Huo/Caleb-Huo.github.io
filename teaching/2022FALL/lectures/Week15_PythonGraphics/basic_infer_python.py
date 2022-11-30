#' ---
#' title: "Introduction to Biostatistical Computing PHC 6937"
#' subtitle: "Basic statistical inference in R"
#' author: Haocheng Ding
#' date: "Wednesday Nov 30th, 2022"
#' output:
#'   slidy_presentation: default
#'   ioslides_presentation: default
#'   beamer_presentation: null
#' ---
#' 

#' 
#' Outline
#' ===
#' 
#' - Simulating random variables
#' - Basic hypothesis testing
#' - linear regression model
#' 
#' 
#' Random samples from a given pool of numbers
#' ===
#' 
import random
a = range(1,11)
random.sample(a, 2)
random.sample(a, 2)
random.sample(a, 2)
#' 
#' Are they really random? Can generate the same random number (for reproducibility)?
#' ===
#' 
#' - The random numbers generated in Python are not truly random, they are **pseudorandom**.
#' 
#' - So we are able to reproduce the exact same random numbers by setting random seeds.
#' 
import random
a = range(1,11)
random.seed(32611) ## 32611 can be replaced by any number.
random.sample(a,2)

random.seed(32611) ## if you keep the same random seed, you will end up with the exact same result.
random.sample(a,2)

random.seed(32608) ## if you update the random seed, you will end up with different result.
random.sample(a,2)
#' 
#' 
#' Each time the seed is set, the same sequence follows
#' ===
#' 
random.seed(32611)
random.sample(range(1,11),2);random.sample(range(1,11),2);random.sample(range(1,11),2)

random.seed(32611)
random.sample(range(1,11),2);random.sample(range(1,11),2);random.sample(range(1,11),2)
#' 
#' Want to sample from a given pool of numbers with replacement
#' ===
#' 
import random
import string

random.sample(range(1,11),10) ## default is without replacement
random.choices(range(1,11),k=10) ## with replacement
random.choices(string.ascii_uppercase,k=10)
#' 
#' Random number generation from normal distribution
#' ===
#' 
#' For normal distribution:
#' 
#' - numpy.random.normal(x, loc=0, scale=1): generate random variable from normal distribution. (rnorm() in R)
#' 
#' from scipy.stats import norm
#' 
#' - scipy.stats.norm.cdf(x, loc=0, scale=1): get CDF/pvalue from normal distribution, $\Phi(x)=(P(Z\leq x))$. (pnorm() in R)
#' 
#' - scipy.stats.norm.pdf(x, loc=0, scale=1): normal density function $\phi = \Phi'(x)$. (dnorm() in R)
#' 
#' - scipy.stats.norm.ppf(q, loc=0, scale=1): quantile function for normal distribution $q(y)=\Phi-1(y)$ such that $\Phi(q(y))=y$. (qnorm() in R)
#' 
#' 
#' Random variable simulators in Python
#' ===
#' 
#' 
#' | Distribution 	| Python command 	|
#' |---	|---	|
#' |Binomial |numpy.random.binomial |
#' |Poisson |numpy.random.poisson |
#' |Geometric |numpy.random.geometric |
#' |Negative binomial |numpy.random.negative_binomial |
#' |Uniform |numpy.random.uniform |
#' |Exponential |numpy.random.exponential |
#' |Normal |numpy.random.normal |
#' |Gamma |numpy.random.gamma |
#' |Beta |numpy.random.beta |
#' |Student t |numpy.random.standard_t |
#' |F |numpy.random.f |
#' |Chi-squared |numpy.random.chisquare |
#' |Weibull |numpy.random.weibull |
#' |Log-normal |numpy.random.lognormal |
#' 
#' 
#' Random variable density in Python
#' ===
#' 
#' | Distribution 	| Python command 	|
#' |---	|---	|
#' |Binomial |scipy.stats.binom.pdf |
#' |Poisson |scipy.stats.poisson.pmf |
#' |Geometric |scipy.stats.poisson.pdf |
#' |Negative binomial |scipy.stats.nbinom.pdf |
#' |Uniform |scipy.stats.uniform.pdf |
#' |Exponential |scipy.stats.expon.pdf |
#' |Normal |scipy.stats.norm.pdf |
#' |Gamma |scipy.stats.gamma.pdf |
#' |Beta |scipy.stats.beta.pdf |
#' |Student t |scipy.stats.t.pdf |
#' |F |scipy.stats.f.pdf |
#' |Chi-squared |scipy.stats.chi2.pdf |
#' |Weibull |scipy.stats.weibull_min.pdf |
#' |Log-normal |scipy.stats.lognorm.pdf |
#' 
#' 
#' Random variable distribution tail probablity in Python
#' ===
#' 
#' | Distribution 	| Python command 	|
#' |---	|---	|
#' |Binomial |scipy.stats.binom.cdf |
#' |Poisson |scipy.stats.poisson.cdf |
#' |Geometric |scipy.stats.poisson.cdf |
#' |Negative binomial |scipy.stats.nbinom.cdf |
#' |Uniform |scipy.stats.uniform.cdf |
#' |Exponential |scipy.stats.expon.cdf |
#' |Normal |scipy.stats.norm.cdf |
#' |Gamma |scipy.stats.gamma.cdf |
#' |Beta |scipy.stats.beta.cdf |
#' |Student t |scipy.stats.t.cdf |
#' |F |scipy.stats.f.cdf |
#' |Chi-squared |scipy.stats.chi2.cdf |
#' |Weibull |scipy.stats.weibull_min.cdf |
#' |Log-normal |scipy.stats.lognorm.cdf |
#' 
#' 
#' Random variable distribution quantile in Python
#' ===
#' 
#' | Distribution 	| Python command 	|
#' |---	|---	|
#' |Binomial |scipy.stats.binom.ppf |
#' |Poisson |scipy.stats.poisson.ppf |
#' |Geometric |scipy.stats.poisson.ppf |
#' |Negative binomial |scipy.stats.nbinom.ppf |
#' |Uniform |scipy.stats.uniform.ppf |
#' |Exponential |scipy.stats.expon.ppf |
#' |Normal |scipy.stats.norm.ppf |
#' |Gamma |scipy.stats.gamma.ppf |
#' |Beta |scipy.stats.beta.ppf |
#' |Student t |scipy.stats.t.ppf |
#' |F |scipy.stats.f.ppf |
#' |Chi-squared |scipy.stats.chi2.ppf |
#' |Weibull |scipy.stats.weibull_min.ppf |
#' |Log-normal |scipy.stats.lognorm.ppf |
#' 
#' 
#' Normal distribution
#' ===
#' 
#' $$f(x;\mu,\sigma)=\cfrac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
#' 
import numpy as np
import matplotlib.pyplot as plt


class Gaussian:
  @staticmethod
  def plot(mean, std, lower_bound=None, upper_bound=None, resolution=None,
    title=None, x_label=None, y_label=None, legend_label=None, legend_location="best"):
    
    lower_bound = ( mean - 4*std ) if lower_bound is None else lower_bound
    upper_bound = ( mean + 4*std ) if upper_bound is None else upper_bound
    resolution  = 100
    
    title        = title        or "Gaussian Distribution"
    x_label      = x_label      or "x"
    y_label      = y_label      or "Density"
    legend_label = legend_label or "μ={}, σ={}".format(mean, std)
    
    X = np.linspace(lower_bound, upper_bound, resolution)
    dist_X = Gaussian._distribution(X, mean, std)
    
    plt.title(title)
    
    plt.plot(X, dist_X, label=legend_label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_location)
    
    return plt
  
  @staticmethod
  def _distribution(X, mean, std):
    return 1./(np.sqrt(2*np.pi)*std)*np.exp(-0.5 * (1./std*(X - mean))**2)
  
plot = Gaussian.plot(0, 1)
plot = Gaussian.plot(1, 1)
plot = Gaussian.plot(0, 2)
plot.show()
#' 
#' t distribution
#' ===
#' 
#' $$f(x;v)=\cfrac{\Gamma(\cfrac{v+1}{2})}{\sqrt{v\pi}\Gamma(v/2)}(1+\cfrac{x^2}{v})^{-\cfrac{v+1}{2}}$$
#' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class Student_t:
  @staticmethod
  def plot(df, lower_bound=None, upper_bound=None, resolution=None,
    title=None, x_label=None, y_label=None, legend_label=None, legend_location="best"):
    
    resolution  = 100
    
    title        =  "Student t Distribution"
    x_label      =   "x"
    y_label      =  "Density"
    legend_label = legend_label or "df={}".format(df)
    
    X = np.linspace(lower_bound, upper_bound, resolution)
    dist_X = Student_t._distribution(X, df)
    
    plt.title(title)
    
    plt.plot(X, dist_X, label=legend_label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_location)
    
    return plt
  
  @staticmethod
  def _distribution(X, df):
    return scipy.stats.t.pdf(X, df)
  
plot = Student_t.plot(2,  lower_bound=-4, upper_bound=4)
plot = Student_t.plot(4,  lower_bound=-4, upper_bound=4)
plot = Student_t.plot(10,  lower_bound=-4, upper_bound=4)
plot = Student_t.plot(float('inf'),  lower_bound=-4, upper_bound=4)
plot.show()
#' 
#' 
#' Chi-square distribution
#' ===
#' 
#' $$f(x;k)=\cfrac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}e^{-x/2}$$
#' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class Chi2:
  @staticmethod
  def plot(df, lower_bound=None, upper_bound=None, resolution=None,
    title=None, x_label=None, y_label=None, legend_label=None, legend_location="best"):
    
    resolution  = 100
    
    title        =  "Chi-square Distribution"
    x_label      =   "x"
    y_label      =  "Density"
    legend_label = legend_label or "df={}".format(df)
    
    X = np.linspace(lower_bound, upper_bound, resolution)
    dist_X = Chi2._distribution(X, df)
    
    plt.title(title)
    
    plt.plot(X, dist_X, label=legend_label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_location)
    
    return plt
  
  @staticmethod
  def _distribution(X, df):
    return scipy.stats.chi2.pdf(X, df)
  
plot = Chi2.plot(1,  lower_bound=0, upper_bound=4)
plot = Chi2.plot(2,  lower_bound=0, upper_bound=4)
plot = Chi2.plot(5,  lower_bound=0, upper_bound=4)
plot = Chi2.plot(10,  lower_bound=0, upper_bound=4)
plt.ylim([0, 1])
plot.show()
#' 
#' 
#' Relationship between Normal distribution and Chi-square distribution
#' ===
#' 
#' - $X\sim~\chi^2_{k}$, where $k$ is the degree of freedom.
#' - $E(X)=2k$.
#' - If $X \sim N(0,1)$, then $X^2\sim\chi^2_1$.
#' 
## verification via qq-plot

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

random.seed(32611)

x1 = np.random.normal(size=1000)
x2 = np.random.chisquare(df=1,size=1000)

x1_sorted = np.sort(np.square(x1))
x2_sorted = np.sort(x2)

plt.xlim([0, 5])
plt.ylim([0, 5])
plt.plot(x1_sorted,x2_sorted,"o")
plot.show()
#' 
#' 
#' Poisson distribution
#' ===
#' 
#' $$f(k:\lambda)=\cfrac{\lambda^ke^{-\lambda}}{k!} (k>0)$$
#' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

x = np.arange(0, 10, 0.5)
# poisson distribution data for y-axis
y1 = scipy.stats.poisson.pmf(x, mu=1)
y2 = scipy.stats.poisson.pmf(x, mu=2)
y3 = scipy.stats.poisson.pmf(x, mu=3)
y4 = scipy.stats.poisson.pmf(x, mu=4)

fig, axs = plt.subplots(2, 2)

plt.setp(axs, ylim=(0,.5)) # set y-axis limit

axs[0, 0].bar(x, y1,color='black')
axs[0, 0].set_title('lambda=1')
axs[0, 1].bar(x, y2,color='red')
axs[0, 1].set_title('lambda=2')
axs[1, 0].bar(x, y3,color='green')
axs[1, 0].set_title('lambda=3')
axs[1, 1].bar(x, y4)
axs[1, 1].set_title('lambda=4')


for ax in axs.flat:
    ax.set(xlabel='x', ylabel='density')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plot.show()
#' 
#' 
#' Beta distribution
#' ===
#' 
#' $$f(x;\alpha,\beta)=\cfrac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}$$
#' 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class Beta:
  @staticmethod
  def plot(a, b, lower_bound=None, upper_bound=None, resolution=None,
    title=None, x_label=None, y_label=None, legend_label=None, legend_location="best"):
    
    resolution  = 100
    
    title        =  "Beta Distribution"
    x_label      =   "Proportion (p)"
    y_label      =  "Density"
    legend_label = legend_label or "a={},b={}".format(a,b)
    
    X = np.linspace(lower_bound, upper_bound, resolution)
    dist_X = Beta._distribution(X, a, b)
    
    plt.title(title)
    
    plt.plot(X, dist_X, label=legend_label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc=legend_location)
    
    return plt
  
  @staticmethod
  def _distribution(X, a, b):
    return scipy.stats.beta.pdf(X, a, b)
  
plot = Beta.plot(a=0.25, b=0.25, lower_bound=0, upper_bound=1)
plot = Beta.plot(a=2, b=2,  lower_bound=0, upper_bound=1)
plot = Beta.plot(a=2, b=5,  lower_bound=0, upper_bound=1)
plot = Beta.plot(a=12, b=2, lower_bound=0, upper_bound=1)
plot = Beta.plot(a=20, b=0.75, lower_bound=0, upper_bound=1)
plot = Beta.plot(a=1, b=1, lower_bound=0, upper_bound=1)
plt.ylim([0, 6])
plot.show()
#' 
#' 
#' Beta distribution properties
#' ===
#' 
#' - $x \sim Beta(\alpha,\beta)$.
#' - $E(X)=\cfrac{\alpha}{\alpha+\beta}$.
#' - $Var(X)=\cfrac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$.
#' - When $\alpha=\beta=1$, Beta distribution reduces to Unif(0,1).
#' 
#' 
#' Samples from Normal distribution
#' ===
#' 
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import linspace
from scipy.stats.kde import gaussian_kde

random.seed(32611)
z = np.random.normal(size=1000) ## generate 1000 samples from N(0,1). 
plt.hist(x=z, bins=50, color='gray',alpha=0.7, rwidth=0.8, density=True)

# plot theoritical pdf
x_axis = np.arange(-4, 4, 0.01)
plt.plot(x_axis, norm.pdf(x_axis, 0, 1),color='red',label='Theoritical')

# plot empirical pdf
kde = gaussian_kde(z)
dist_space = linspace( min(z), max(z), 100 )
plt.plot(dist_space, kde(dist_space), color='#104E8B',label='Empirical')

plt.legend(loc="upper left")
plt.grid(axis='y', alpha=0.75)
plt.xlabel('z')
plt.ylabel('Density')
plt.title('Histogram of z')
plt.show()
#' 
#' 
#' Check CDF
#' ===
#' 
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF

random.seed(32611)
z = np.random.normal(size=1000) ## generate 1000 samples from N(0,1).



# plot theoritical cdf
x = np.linspace(-3, 3, 1000)
y = scipy.stats.norm.cdf(x)
plt.plot(x, y,label='Theoritical distribution')

# plot empirical distribution
ecdf = ECDF(z)
plt.plot(ecdf.x, ecdf.y, label='Empirical distribution')

plt.legend(loc="upper left")
plt.xlabel('x')
plt.ylabel('Distribution')
plt.title('Empirical distribution')
plt.show()
#' 
#' 
#' p-value
#' ===
#' 
#' - p-value: The probability that the null statistic is as extreme or more extreme than the observed statistic, when null hypothesis is true.
#' 
#' - p-value is the cumulative density. It can be computed by scipy.stats.norm.cdf (Normal distribution) in python.
#' 
#' - Example: observed Z value equals to 2 with null distribution is $N(0,1)$.
import scipy.stats

2*(1-scipy.stats.norm.cdf(2)) ## Two-sided test

1-scipy.stats.norm.cdf(2) ## One-sided test
#' 
#' 
#' Quantile
#' ===
#' 
#' scipy.stats.norm.ppf(): return the quantile of input area under the density curve (cumulative density). 
#' 
import scipy.stats

scipy.stats.norm.ppf(0.975)

scipy.stats.norm.ppf(.025)

scipy.stats.norm.ppf(.5)
#' 
#' Commonly used statistical tests
#' ===
#' 
#' - T test
#' - Wilcox rank test
#' - Chi-squre test
#' - Fisher's exact test
#' - Correlation test
#' - KS test
#' 
#' 
#' One-sample T test
#' ===
#' 
#' - Test if the group mean is different from 0.
#' - $H_0$: mean value of a group is 0 ($\mu$=0).
#' 
import random
import numpy as np
import scipy.stats

random.seed(32611)
group = np.random.normal(size=10)

scipy.stats.ttest_1samp(group,popmean=0)

#' 
#' Two-sample T test
#' ===
#' 
#' - To compare the mean values of two samples that are normally distributed.
#' - $H_0$: mean values of two samples are same ($\mu_1=\mu_2$). 
import random
import numpy as np
import scipy.stats

random.seed(32611)
group1 = np.random.normal(size=10)
group2 = np.random.normal(size=10,loc=1.2)

scipy.stats.ttest_ind(group1,group2)
#' 
#' 
#' Two-sample paired T test
#' ===
#' 
#' - When your data are in matched pairs. 
#' - $H_0$: mean values of two samples are same ($\mu_{pre}=\mu_{post}$). 
#' 
import random
import numpy as np
import scipy.stats

random.seed(32611)
pre = np.random.normal(size=10)
post = np.random.normal(size=10,loc=1.2)

scipy.stats.ttest_rel(pre,post)
#' 
#' 
#' 
#' Two-sample Wilcox Rank Sum test
#' ===
#' 
#' - To compare the median values of two samples without any distribution assumption.
#' 
#' - Non-parametric test.
#' 
#' - $H_0$: median values of two samples are same.
#' 
import random
import numpy as np
import scipy.stats

random.seed(32611)
group1 = np.random.normal(size=10)
group2 = np.random.normal(size=10,loc=1.2)

scipy.stats.ranksums(group1, group2)
#' 
#' 
#' Two-sample paired Wilcox Rank Sum test
#' ===
#' 
#' - When your data are in matched pairs.
#' 
#' - $H_0$: median values of two samples are same.
#' 
#' 
import random
import numpy as np
import scipy.stats

random.seed(32611)
group1 = np.random.normal(size=10)
group2 = np.random.normal(size=10,loc=1.2)

scipy.stats.wilcoxon(group1, group2)
#' 
#' 
#' Chi-square test
#' ===
#' 
#' 

#' 
#' - Test for independence
#' - $H_0$: the treatment effects are same between treatment group and control group. 
#' - Note that:
#'   - All expected counts should be greater than 5.
#'   - Can be applied to tables with more than 2 categories.
#' 
#' - Example: relationship between myocardial infarction and aspirin use (Agresti 1996).
#' 
#' 

#' 
#' 
from scipy.stats import chi2_contingency

observed = [[189, 10845], [104,10933]]
chi2_contingency(observed)

## Output details
#(test statistic, p-value, df, expected frequencies)
#' 
#' 
#' Fisher's exact test
#' === 
#' 
#' - Test for independence.
#' - No approximation. (When more than 20% of cells have expected frequencies < 5.)
#' - $H_0$: the treatment effects are same between treatment group and control group.
#' 
#' 

#' 
#' 
from scipy.stats import fisher_exact

observed = [[189, 10845], [104,10933]]
scipy.stats.fisher_exact(observed)

## Output details
#(odds ratio, p-value)
#' 
#' 
#' Correlation test
#' ===
#' 
#' - Test if there is a correlation between vector a and b.
#' - $H_0: \rho_1=\rho_2$.
#' 
import random
import numpy as np
import scipy.stats

random.seed(32611)
group1 = np.random.normal(size=10)
group2 = np.random.normal(size=10,loc=1.2)

scipy.stats.pearsonr(group1,group2) # Pearson correlation: requires Gaussian assumption.

scipy.stats.spearmanr(group1,group2) # Spearman correlation: non-parametric, no distribution assumption needed.

#' 
#' Kolmogorov-Smirnov (KS) test
#' ===
#' 
#' - Test if random variables X and Y are from the same distribution.
#' 
#' - $H_0$: X and Y are from the same distribution.
#' 
import random
import numpy as np
import scipy.stats

random.seed(32611)
group1 = np.random.normal(size=50) # N(0,1)
group2 = np.random.normal(size=50,loc=1.2) # N(1.2,1)


scipy.stats.kstest(group1, group2)
#' 
#' Fit linear model in Python 
#' ===
#' 
#' - sklearn
#' 
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd 

mtcars = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv", index_col=0)

#Setting response and predictors
y = mtcars.mpg
X = mtcars[["cyl"]]

#Fitting simple Linear Regression Model
linr_model = LinearRegression().fit(X, y)

#Model Fitting Results
print('estimated slope (cly)', 'estimated intercept', 'R-square')
print(linr_model.intercept_, linr_model.coef_[0],linr_model.score(X,y)) 
#' 
#' - Interpretation
#'   - One unit increase in cyl will result in 2.88 decrease in mpg on average.
#'   - 0.726 denoted that 72.6% of the variability is explained by the model, which means the predicted values are 72.6% close to the actual mpg values. 
#' 
#' 
#' -------------------------------------------------------------
#' 
#' - Making Predictions based on the estimated slope and intercept.
#'   - Cylinders.
mtcars[["cyl"]].head(10)
#' 
#'   - Predicted mpgs. 
linr_model.predict(mtcars[["cyl"]])[0:10]
#' 
#' 
#' -------------------------------------------------------------
#' 
#' - statsmodels
#' 
import statsmodels.api as sm

#define response variable
y = mtcars['mpg']

#define predictor variables
x = mtcars[['cyl']]

#add constant to predictor variables
x = sm.add_constant(x)

#fit linear regression model
model1 = sm.OLS(y, x).fit()

#view model summary
print(model1.summary())
#' 
#' -------------------------------------------------------------
#' 
#' - Response: mpg.
#' - Predictors: cyl, disp.
#' 
y = mtcars['mpg']
x = mtcars[['cyl','disp']]
x = sm.add_constant(x)
model2 = sm.OLS(y, x).fit()
print(model2.summary())
#' 
#' -------------------------------------------------------------
#' 
#' - Response: mpg.
#' - Predictors: all variables but the response (mpg).
#' 
y = mtcars['mpg']
x = mtcars.loc[:, mtcars.columns != 'mpg'] # exclude mpg
x = sm.add_constant(x)
model3 = sm.OLS(y, x).fit()
print(model3.summary())
