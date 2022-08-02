import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression  #used for the example of least squares line only
from sklearn.linear_model import RANSACRegressor #used for example of 'something else' line only
import statsmodels.api as sm
import statsmodels.formula.api as smf

'''Problem 1'''
# We would like to determine if the average length of stay can predict the average hospital charge 
# and use it to make a prediction for the charge when a patient stays 4 days. Perform a regression 
# analysis to determine the answer to this question, use α = 0.10 when required.

#Read the data and assign 'x' and 'y'
hospital=pd.read_csv('hospital_data.csv')
hospital = hospital.rename(columns = {'Average Charge': 'avg_charge', 'Average Length of Stay (days)': 'avg_len_stay'})
x = hospital['avg_len_stay']
y = hospital['avg_charge']

#Reset the plot so that I can use it as a blank slate to add to later
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, alpha=0.9, color='orchid')
fig.suptitle('Hospital predictions')
plt.xlabel("Average Length of Stay (days)")
plt.ylabel("Average Charge")
fig.tight_layout(pad=2); 
ax.grid(True)
# fig.savefig('hospital.png', dpi=300)

#Calculate some basic values for use later
xbar=np.mean(x)
xsum=sum(x)
ybar=np.mean(y)
ysum=sum(y)
xy=np.multiply(x,y)
xx=np.multiply(x,x)
sumxy=sum(xy)
sumxx=sum(xx)
n=len(x)

#Calculate betas and fitted line
SSxy=sumxy-xsum*ysum/n
SSxx=sumxx-xsum**2/n
beta1=SSxy/SSxx
beta0=ybar-beta1*xbar
fitted = lambda xx: beta0 + beta1*xx  

#Create a plot with the fitted line
x_pred = np.linspace(x.min(), x.max(), 50)
y_pred=fitted(x_pred)
ax.plot(x_pred, y_pred, '-', color='darkblue', linewidth=2)
# fig.savefig('propellantfit.png', dpi=300)

#More calculatsions, this time for error
SSyy=sum(np.multiply(y,y))-ysum**2/n
SSE=SSyy-beta1*SSxy
MSE=SSE/(n-2)
s=MSE**(1/2)

alpha = 0.1

r = SSxy/(SSxx*SSyy)**(1/2)
r2 = r**2
r2alt = 1-SSE/SSyy

tstatr = r*(np.sqrt(n-2))/np.sqrt(1-r**2)
pval = stats.t.sf(tstatr, n-2)
print('tstat = ', tstatr)
print('pval = ', pval)

#Confidence interval around Beta and t-test for Beta
tstat=beta1/(s/(SSxx**(1/2)))
tcrit=stats.t.ppf(1-0.05/2,n-2)
UL=beta1+tcrit*s/(SSxx**(1/2))
LL=beta1-tcrit*s/(SSxx**(1/2))

print('We are', 100*(1-alpha), '%confident that the true value of the slope is in the interval', round(LL,2),',',round(UL,2),')')

#Estimation and Prediction Interval calculations
se_est = lambda x: s * np.sqrt(  1./n + (x-xbar)**2/SSxx)
se_pred = lambda x: s * np.sqrt(1+1./n + (x-xbar)**2/SSxx)

#Sample calculations for x_p=4
x_p=4
UpperEstEx=fitted(x_p)+abs(stats.t.ppf(1-alpha/2,n-2)*se_est(x_p))
LowerEstEx=fitted(x_p)-abs(stats.t.ppf(1-alpha/2,n-2)*se_est(x_p))
UpperPredEx=fitted(x_p)+abs(stats.t.ppf(1-alpha/2,n-2)*se_pred(x_p))
LowerPredEx=fitted(x_p)-abs(stats.t.ppf(1-alpha/2,n-2)*se_pred(x_p))


### Visualizing Intervals
upper = y_pred + abs(stats.t.ppf(1-alpha/2,n-2)*se_est(x_pred))
lower = y_pred - abs(stats.t.ppf(1-alpha/2,n-2)*se_est(x_pred))
ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.6)
upperp = y_pred + abs(stats.t.ppf(1-0.05/2,n-2)*se_pred(x_pred))
lowerp = y_pred - abs(stats.t.ppf(1-0.05/2,n-2)*se_pred(x_pred))
ax.fill_between(x_pred, lowerp, upperp, color='#888888', alpha=0.2)
# fig.savefig('CIex.png', dpi=300)

#output Model Summary
df=pd.DataFrame({'length':x,'charge':y})
model1=smf.ols('charge~length',df).fit()
print(model1.summary2(alpha=alpha))

#calculate prediction intervals
dfpred = pd.DataFrame({'length':[x_p,10000],'charge':[fitted(x_p),fitted(10000)]})
prediction=model1.get_prediction(dfpred)
predints=prediction.summary_frame(alpha=alpha)

anova_table = sm.stats.anova_lm(model1)
anova_table[:-1]['sum_sq']/sum(anova_table['sum_sq'])
