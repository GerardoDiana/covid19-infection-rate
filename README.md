# covid19-infection-rate
Using data acquired by COVID-19 Data Repository by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU) and Our World in Data COVID-19 dataset. Developed hierarchical models to analyze and predict the rate of infection in the U.S.
- Data contains the above collected data
- CovidCountModel.ipynb: bayesian hierarchical model with the following structure,
\begin{align*}
y_i &\sim Pois(\lambda_i)\\
\lambda_i &\sim Gamma(\alpha,\beta)\\
\beta &\sim Gamma(a,b)
\end{align*}
- CovidCountModel2.ipynb: bayesian hierarchical model with a structure to be determined
  
