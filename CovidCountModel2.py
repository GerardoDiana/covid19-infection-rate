#!/usr/bin/env python
# coding: utf-8

# ## Model 2 For Coivd-19 Data

# In[84]:


# import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import os
import seaborn
from scipy.stats import poisson


# In[39]:


# Get current working directory
print(os.getcwd())


# In[40]:


# Read in the data
usa_covid = pd.read_csv("Data/usa_covid_cases.csv")
# get rid of the index column (optional)
usa_covid.reset_index(drop=True, inplace=True)

print(usa_covid.shape)


# In[41]:


# Organize usa_covid to have only:
# the date, new cases, and icu patients columns
usa_covid_subset = usa_covid[['date', 'new_cases', 'icu_patients']]
print(usa_covid_subset)


# In[42]:


# Check for Nas
print("# of NAs in date: ", usa_covid_subset['date'].isna().sum())
print("# of NAs in new_cases: ", usa_covid_subset['new_cases'].isna().sum())
print("# of NAs in icu_patient: ",usa_covid_subset['icu_patients'].isna().sum())


# In[43]:


# Now I have the final data to use for model 2
final_covid_df = usa_covid_subset.dropna()
print(final_covid_df)


# In[44]:


# histogram plot of ICU patients 
seaborn.distplot(final_covid_df['icu_patients'],kde=False, color = "purple", hist_kws={'edgecolor': 'black'}
                ).set(xlabel='ICU patients', ylabel= 'total number', title='Histogram of ICU Patients')
plt.show()


# In[45]:


# density plot of new covid19 cases
seaborn.distplot(final_covid_df['new_cases'], hist = False, color = 'purple'
                ).set(title='Density of New Covid-19 Cases', ylabel='Density', xlabel='new cases')
plt.show()


# ## Model 2 Specifications
# Let $y_i$ be the count of new covid-19 cases for day $i$ and let $c_i$ be the ICU patient count for day $i$. I assume that $y_i \sim \text{Bin}(N_i,\,\theta_i)$, which corresponds to $y_i$ the observed count of new covid-19 cases, out of a population of $N_i$ new covid-19 cases. The probability of observing a new covid-19 case is $\theta_i$. Unforunately, $N_i$ is not known. So, I assume that $N_i \sim \text{Pois}(\lambda_{i}c_i)$. To complete the model I assume that $\lambda_i \sim \text{Ga}(\alpha_\lambda\,,\beta_\lambda)$ and that $\theta_i \sim \text{Be}(1,\beta_\theta)$. Thus, the joint posterior is:
# \begin{align*}
# p(N_i,\lambda_i,\theta_i|y_i) &\propto \prod_{i=1}^n f(y_i|N_i,\theta_i)\, \pi(N_i|\lambda_{i}c_i)\, \pi(\lambda_i|\alpha_\lambda,\beta_\lambda)\,\pi(\theta_i|\beta_\theta) \\
# &\propto \prod_{i=1}^n \frac{N_i!}{y_i!(N_i-y_i)!}\theta_i^{y_i}(1-\theta_i)^{N_i - y_i}\frac{(\lambda_{i}c_i)^{N_i}\text{exp}(-\lambda_{i} c_i) }{N_{i}!} \frac{\beta_{\lambda}^{\alpha_\lambda}}{\Gamma(\alpha_\lambda)}\lambda_{i}^{\alpha_\lambda - 1}\text{exp}(-\lambda_i \beta_\lambda)\frac{\theta_i^{1-1}(1-\theta_i)^{\beta_{\theta}-1}}{\text{B}(1,\beta_\theta)} \\
# &\propto \prod_{i=1}^n \frac{1}{y_i!(N_i-y_i)!}\theta_i^{y_i}(1-\theta_i)^{N_i - y_i}(\lambda_{i}c_i)^{N_i}\text{exp}(-\lambda_{i} c_i) \lambda_{i}^{\alpha_\lambda - 1}\text{exp}(-\lambda_i \beta_\lambda)(1-\theta_i)^{\beta_{\theta}-1}
# \end{align*}

# From this, I obtain the full conditionals:
# \begin{align*}
# (N_i|\cdot) &\propto \frac{1}{(N_i - y_i) !}(1-\theta_i)^{N_i - y_i}(\lambda_i c_i)^{N_i} \\
# &\\
# (\lambda_i|\cdot) &\propto (\lambda_i c_i)^{N_i}\text{exp}(-\lambda_i c_i)\lambda_i^{\alpha_\lambda - 1}\text{exp}(-\lambda_i \beta_\lambda) \\
# &\propto (\lambda_i)^{(N_i+\alpha_\lambda)-1}\text{exp}\left\{-\lambda_i(c_i + \beta_\lambda)\right\}\\
# &\sim \text{Ga}(N_i+\alpha_\lambda,\,c_i + \beta_\lambda) \\
# &\\
# (\theta_i|\cdot) &\propto \theta_i^{y_i}(1-\theta_i)^{N_i - y_i}(1-\theta_i)^{\beta_\theta - 1}\\
# &\propto \theta_i^{y_i}(1-\theta_i)^{N_i - y_i + \beta_\theta - 1}\\
# &\sim \text{Be}(y_i,\, N_i - y_i + \beta_\theta)
# \end{align*}

# Note that the full conditional for $N_i|\cdot$ seems to follow a Poisson distribution with an offset. So our proposal density will follow a Poisson

# In[91]:


def ln(n):
    return(np.log(n))
def ln_factorial(n):
    # """ returns upper bound of sterlings approxiamtion """
    # assert n >= 0, "n must be positive"
    upp_bound = n*ln(n)
    return upp_bound


# In[92]:


# define the natural log of the full conditional Ni
# may return a series instead of an array
def Ni_ln_full_conditional(Ni, yi, ci, theta_i, lambda_i):
    return (-ln_factorial(Ni-yi) + (Ni-yi)*ln(1-theta_i) + Ni*ln(lambda_i * ci))


# In[93]:


def accept_prob_function(NewCond, CurrCond, NewProp, CurrProp):
    value = np.exp(NewCond - CurrCond) * np.exp(CurrProp - NewProp)
    return(value)


# In[116]:


def metropolis_hastings(N_mcmc, yi, ci, a_lam, b_lam, b_theta):
    n = len(yi)
    
    # initialize your samples
    lambda_prior = np.random.gamma(shape = a_lam, scale = 1/b_lam, size = n)
    Ni_current_proposal = np.random.poisson(lam = lambda_prior*ci, size = n)
    
    lambda_i = np.zeros(shape = (N_mcmc,n) )
    theta_i = np.zeros(shape = (N_mcmc,n) )
    
    # Note Ni_samps will be our final Ni samples from the algorithm
    Ni_samps = np.zeros(shape = (N_mcmc,n))
    accept_prob = np.zeros(shape = n)
    count_accept = 0
    count_reject = 0
    
    for i in range(N_mcmc):
        # gibbin': iteratively update lambda_i, theta_i, Ni_new_proposal
        lambda_i[i,:] = np.random.gamma(shape = Ni_current_proposal + a_lam, scale = 1/(ci + b_lam), size = n)
        theta_i[i,:] = np.random.beta(a = yi, b = Ni_current_proposal + b_theta, size = n)
        Ni_new_proposal = np.random.poisson(lam = lambda_i[i,:]*ci, size = n)
    
        # Metropolis-Hastings for Ni
        Ni_curr_full_conditional = Ni_ln_full_conditional(Ni_current_proposal, yi, ci, theta_i[i,:], lambda_i[i,:])
        Ni_new_full_conditional = Ni_ln_full_conditional(Ni_new_proposal, yi, ci, theta_i[i,:], lambda_i[i,:])
        
        Ni_curr_logpmf = poisson.logpmf(Ni_current_proposal, lambda_i[i,:]*ci, loc=0)
        Ni_new_logpmf = poisson.logpmf(Ni_new_proposal, lambda_i[i,:]*ci, loc=0)
    
        for j in range(n):
            accept_prob[j] = accept_prob_function(
                Ni_new_full_conditional.values[j], Ni_curr_full_conditional.values[j], 
                Ni_new_logpmf[j], Ni_curr_logpmf[j]
                )
            q = np.minimum(1, accept_prob[j])
            
            if np.random.uniform(0,1,1) <= q:
                        Ni_samps[i,j] = Ni_new_proposal[j]
                        count_accept = count_accept + 1    
            else: 
                        Ni_samps[i,j] = Ni_current_proposal[j]
                        count_reject = count_reject + 1
    
    return(lambda_i,theta_i, Ni_samps)


# In[117]:


lambda_i,theta_i,Ni_samps = metropolis_hastings(N_mcmc = 10000, yi = final_covid_df['new_cases'],
                                                ci = final_covid_df['icu_patients'], 
                                                a_lam = 10000, b_lam = 10000/1000, b_theta = 500)


# In[ ]:
# for numpy.delete(), axis= 0 implies row, 1 implies column
# obj: the row/column number 
posterior_lambda= np.delete(lambda_i, obj= range(6000), axis= 0)
posterior_theta= np.delete(theta_i, obj= range(6000), axis=0 )
posterior_Ni = np.delete(Ni_samps, obj= range(6000), axis=0)

print("Dimensions of Posterior Lambda: ", posterior_lambda.shape)
print("Dimensions of Posterior Thetha: ", posterior_theta.shape)
print("Dimensions of Posterior Ni: ", posterior_Ni.shape)
print("Length of Posterior Lambda: ", len(posterior_lambda[:,1]))

# In[]:
plt.plot(posterior_Ni[:,0])