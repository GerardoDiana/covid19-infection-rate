#!/usr/bin/env python
# coding: utf-8

# ### Count Model for Covid-19 Data
# 
# ### First import/upload the common packages:
# - pandas: create, read, manipulate dataframes
# - numpy: perform matrix and array operations
# - matplotlib.pyplot: creates plots

# In[1]:


import pandas as pd                  # pd short for pandas
import numpy as np                   # np short for numpy
import matplotlib.pyplot as plt     # plot short for matplotlib.pyplot
from matplotlib import style
style.use('ggplot')


# Few keyboard shortcuts
# - alt + enter (creates new code cell)
# - esc + m (changes to markdown)
# - esc + y (changes to code)
# - ctrl + / (comment out multiple lines)
# 
# ### Read in the data
# Our *usa_covid_cases.csv* is not in the "same" folder as this pyscript. To view the current working directory you must:

# In[2]:


import os
print(os.getcwd())     # get current working directory


# We see that the above is our current working directory, and the data we need is located in a subfolder in Covid-CountModel called Data. So the only part of the path we state is:

# In[3]:


usa_rona = pd.read_csv("Data/usa_covid_cases.csv")

# get rid of the index column (optional)
usa_rona.reset_index(drop=True, inplace=True)


# ### Preview the data
# Previewing the data is similar to Rstudio with the following:

# In[4]:


print(usa_rona.shape) # dataframe dimensions (rows,columns)
print(usa_rona.head()) # view first 5 rows


# ### View Column Names 
# To properly view the 59 column names we will print a list of the "columns.values" 

# In[5]:


print(list(usa_rona.columns.values))


# ### Create New Dataframe Containing the Columns/Variables of Interest
# Relevant variables are the "date" and "new cases". In general to extract/view certain columns use $\text{data}[[\text{ 'A', 'B' }]]$.

# In[6]:


usa_rona_subset = usa_rona[['date', 'new_cases']]
print(usa_rona_subset)


# ### Discover and Remove any Nans in the data
# To find the number of Nas, we will first specify which column in the data and use .isna().sum()

# In[7]:


print(usa_rona_subset['date'].isna().sum())
print(usa_rona_subset['new_cases'].isna().sum())


# To remove the Nas we use '.dropna()'

# In[8]:


final_covid_df = usa_rona_subset.dropna()
print(final_covid_df)


# ### Exploratory Data Analysis

# In[9]:


final_covid_df.plot(x = 'date', y = 'new_cases', color = 'purple')
plt.xlabel('date')     # 2020-01-22 to  2021-02-23
plt.xticks(rotation = 45)
plt.ylabel('new cases')
plt.title('New cases VS Date')
plt.show()


# In[10]:


# using matplotlib.pyplot to make a historgram

plt.hist(final_covid_df['new_cases'], color = 'darkorchid', edgecolor = 'black',
        bins = 20)
plt.xlabel('new cases')
plt.ylabel('total number')
plt.title('Histogram of New USA Covid Cases')
plt.show()


# In[11]:


# instead of matplot you can use seaborn
import seaborn 

# Density Plot and Histogram of new usa covid cases
# We can call the den and hist in one fucntion
seaborn.distplot(final_covid_df['new_cases'], hist=True, kde=True, bins=20, 
                 color = 'purple', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}
                ).set(xlabel='newcases',ylabel='Density',title='Density and Histogram of New USA Covid Cases')
plt.xticks(rotation = 45)
plt.show()


# In[12]:


# plot of just the histogram
seaborn.distplot(final_covid_df['new_cases'], hist=True, kde = False,
                 bins=20, color = 'purple', hist_kws={'edgecolor':'black', 'linewidth':1.5}
                ).set(xlabel='new cases', ylabel='total number', title = 'Histogram of New USA Covid Cases')
plt.show()


# In[195]:


# plot of just the density
seaborn.distplot(final_covid_df['new_cases'], hist = False, kde = True,
                 bins = 20, kde_kws = {'shade': True, 'linewidth': 2, 'color': 'purple'}
                ).set(xlabel='new cases', ylabel='Density', title='Density of New USA Data Covid Cases')
plt.xticks(rotation = 45)
plt.show()


# ## Model 1 Specification
# We assume $y_i\sim\text{Pois}(\lambda_i)$ which corresponds to $y_i$ the observed count of new covid-19 cases where $\lambda_i$ is the unknown rate of catching new covid-19 cases. To complete the model we assume 
# $$y_i\sim\text{Pois}(\lambda_i)\\
# \lambda_i \sim\text{Ga}(\alpha,\theta)\\
# \theta\sim\text{Ga}(a,b)$$
# where $\alpha,a,b$ are fixed. The joint posterior distribution becomes
# 
# \begin{align*}
# p(\lambda,\theta|y) &\propto\left[\prod_{i=1}^n \frac{\lambda_{i}^{y_i} \text{exp}\left\{-\lambda_i\right\}}{y_i!}
# \frac{\lambda_{i}^{\alpha}}{\Gamma(\alpha)}\theta^{\alpha-1} \text{exp}\left\{-\theta\lambda_i\right\}  \right] \frac{\theta^a}{\Gamma(a)} b^{a-1} \text{exp}\left\{-b\theta\right\}\\
# &\propto \left[\prod_{i=1}^n \lambda_{i}^{y_i} \text{exp}\left\{-\lambda_i\right\} \lambda_{i}^{\alpha} \theta^{\alpha-1}  \text{exp}\left\{-\theta\lambda_i\right\}\right] \theta^a \text{exp}\left\{-b\theta\right\}
# \end{align*}
# 
# The distribution in brackets is going to follow a gamma distribution because the conjugate prior to a Poisson distribution is a Gamma Prior thus the conjugate posterior will follow a gamma distribution aswell. Thus the full conditionals are
# 
# \begin{align*}
# (\lambda_i|\theta,y_i) &\propto \lambda_{i}^{y_i} \text{exp}\left\{-\lambda_i\right\} \lambda_{i}^{\alpha} \text{exp}\left\{-\theta\lambda_i\right\} \\
# &\propto \lambda_{i}^{y_i + \alpha} \text{exp}\left\{-\lambda_i(\theta + 1)\right\} \\
# &\sim \text{Ga}(y_i +\alpha,\, \theta + 1) \\
# (\theta|\lambda_i,y_i) &\propto \left[\prod_{i=1}^n \theta^{\alpha-1}\text{exp}\left\{-\theta\lambda_i \right\}\right]\theta^a \text{exp}\left\{-b\theta\right\} \\
# &\propto \theta^{\alpha n - n}\text{exp}\left\{-\theta \sum_{i=1}^{n} \lambda_i\right\}\theta^a \text{exp}\left\{-b\theta\right\} \\
# &\propto \theta^{\alpha n - n + a}\text{exp}\left\{-\theta (b+\sum_{i=1}^{n} \lambda_i)\right\} \\
# &\sim \text{Ga}\left(\alpha n +a - (n-1),\, b+\sum_{i=1}^{n} \lambda_i \right)
# \end{align*}
# 
# ### Side Note: How to Sample From Dists and Initialize Vectors

# In[14]:


# Example: how to sample from a distribution using numpy
# NOTE: we are using the rate parameter not scale! so scale = 1/beta
s = np.random.gamma(shape = 1 , scale = 1/2 , size = 1000)
seaborn.distplot(s, hist=True, kde=True, bins=30, color = 'royalblue', 
                 hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}
                ).set(xlabel = 'x', ylabel='Density', Title = 'title')
plt.show()


# In[15]:


# Example: how to initialize the vectors to store your samples
# in R: sample = matrix(NA,nrow = N_mcmc, ncol = length(obs))
# in Python:
sample = np.zeros(shape=(100,20)) # use zeros instead of NA to tell python we're gonna fill this with floats
print("Initialized sample \n", sample)


# In[16]:


print(sample.shape)


# In[17]:


# so at each iteration you want to replace with your random sample
# reminder: python indexes at zero
# ex: range(5) = (0,1,2,3,4)
for i in range(2):
    sample[i,:] = np.random.normal(loc=0, scale=1, size=20)
print("New Sample: \n",sample)
print("The Sum of First Row: \n" ,np.sum(sample[0,]))
sample[0,:] = 1
print(sample)


# In[18]:


# Python equivalent to: sample = rep(NA, 100)
sample2 = np.zeros(shape=100)
print(sample2)
for i in range(3):
    # size = 1 because we're getting only one sample at each iteration not a vector
    sample2[i] = np.random.normal(loc=0, scale=1, size=1)
print(sample2)


# ## Back To our Problem: Defining our Gibbs Sampler

# In[172]:


# Defin a function for Gibbs Sampling
# N_mcmc: number of iters
def Gibbs(N_mcmc, yi, alpha, a, b): 
    n = len(yi)

    # initialize your samples
    lambda_i = np.zeros(shape=(N_mcmc,n))
    theta = np.zeros(shape=N_mcmc)
    
    lambda_i[0,:] = 1
    theta[0] = 1
    
    for i in range(1,N_mcmc):
        # iteratively update lambda_i and theta
        lambda_i[i,:] = np.random.gamma(shape= yi+alpha, scale= 1/(theta[i-1]+1), size= n)
        theta[i] = np.random.gamma(shape= alpha*n + a, scale= 1/(b + np.sum(lambda_i[i,:])), size= 1)        
    
    return(lambda_i, theta)
# - (n-1)


# In[173]:


lambda_i,theta = Gibbs(N_mcmc=10000, yi = final_covid_df['new_cases'], alpha= 100, a=1, b=1 )
print("Lamda_i Samples \n", lambda_i)
print("Theta Samples \n", theta)


# ### Removing the First 6000 iterations from the Posterior Parameters

# In[174]:


# for numpy.delete(), axis= 0 implies row, 1 implies column
# obj: the row/column number 
posterior_lambda= np.delete(lambda_i, obj= range(6000), axis= 0)
posterior_theta= np.delete(theta, obj= range(6000) )

print("Dimensions of Posterior Lambda: ", posterior_lambda.shape)
print("Dimensions of Posterior Thetha: ", posterior_theta.shape)
print("Length of Posterior Lambda: ", len(posterior_lambda[:,1]))


# # Make Trace Plots!

# In[175]:


plt.plot(posterior_lambda[:,4])


# In[165]:


plt.plot(posterior_theta[:])


# ### Plotting the Full Posterior Conditionals

# In[176]:


# with numpy.mean(), axis= 0 implies columns, 1 implies row 
avg_lambda= np.mean(posterior_lambda, axis=0)
avg_lambda.shape


# In[177]:


seaborn.distplot(avg_lambda, hist = True, kde = True, bins=20,
                color='brown', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}
                ).set(xlabel= '$\lambda_{avg}$', ylabel='Density',
                      title='Density and Histogram of $\lambda_{avg}$')
plt.show()


# In[178]:


seaborn.distplot(posterior_lambda[3400,], hist = True, kde = True, bins=20,
                color='brown', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2}
                ).set(xlabel= '$\lambda_{i}$', ylabel='Density',
                      title='Density and Histogram of $\lambda_{i}$')
plt.show()


# In[179]:


seaborn.distplot(posterior_theta, hist=True, kde=True, color='brown',
                hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth':2}
                ).set(xlabel='$\u03B8$')
plt.show()


# ## The Predictive Posterior Distribution
# Let $\hat{y}_i$ denote the predictive posterior distribution. We wish to predict the outcome of new postitive covid-19 cases, $\hat{y}_i$. One simulates $\hat{y}_i$ by:
# - simulating $(\lambda_i,\,\theta)$ from the joint posterior given $y_i$
# - simulating $\hat{y}_i$ from its sampling density given the simulated values of $\lambda_i$ and $\theta$, $\hat{y}_i\sim\text{Pois}(\lambda_i)$

# In[180]:


# Define a function for the posterior predictive
def poisson_posterior_predictive(lam_samps): 
    n = len(lam_samps[0,:])
    N_mcmc = len(lam_samps[:,0])
    
    posterior_predictive = np.zeros(shape=(N_mcmc,n))
    
    for i in range(N_mcmc):
        posterior_predictive[i,:] = np.random.poisson(lam = lam_samps[i,:], size = n)
    return(posterior_predictive)


# In[181]:


# Simuate from the posterior predictive distribution
posterior_predictive = poisson_posterior_predictive(posterior_lambda)
print(posterior_predictive.shape)


# ### Posterior Predictive Check
# Notice the model captures the prior information more than the observed data. So we would have to propose a better model that better captures the observed data

# In[183]:


quantile_posterior_predictive = np.quantile(posterior_predictive, [.025, .5, .97], axis=0)
print(quantile_posterior_predictive.shape)


# In[184]:


def plot_mean_and_CI(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, lw = .3)


# In[188]:


# zoomed between days 70 - 80 because its hard to see that damn shaded interval
mean = quantile_posterior_predictive[1,:]
lower = quantile_posterior_predictive[0,:]
upper = quantile_posterior_predictive[2,:]

plot_mean_and_CI(mean, upper,lower, color_mean='b', color_shading='k')


# In[191]:


plt.plot(final_covid_df['new_cases'])
plt.plot(posterior_predictive.mean(0))


# In[192]:


final_covid_df['new_cases'] - posterior_predictive.mean(0)


# In[196]:


# lets try to do the one with the lines and the dots
CI_posterior_predictive = np.quantile(posterior_predictive, [.025, .97], axis=0)
index = len(CI_posterior_predictive[0,:])

