import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression  #used for the example of least squares line only
from sklearn.linear_model import RANSACRegressor #used for example of 'something else' line only
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

#ignore by message
warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20")

'''Problem 1'''
# We would like to determine if the average length of stay (x) can predict the average hospital charge (y)
# and use it to make a prediction for the charge when a patient stays 4 days. Perform a regression 
# analysis to determine the answer to this question, use α = 0.10 when required.

#Read the data and assign 'x' and 'y'
hospital=pd.read_csv('hospital_data.csv')
hospital = hospital.rename(columns = {'Average Charge': 'avg_charge', 
                            'Average Length of Stay (days)': 'avg_len_stay'})
x = hospital['avg_len_stay']
y = hospital['avg_charge']
alpha = 0.10

#Calculate some basic values for use later
def linear_regression(x, y):     
    N = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    B1_num = ((x - x_mean) * (y - y_mean)).sum()
    B1_den = ((x - x_mean)**2).sum()
    B1 = B1_num / B1_den
    
    B0 = y_mean - (B1 * x_mean)
    
    reg_line = 'y = {} + {}β'.format(B0, round(B1, 4))
    
    return (B0, B1, reg_line)

def corr_coef(x, y):
    N = len(x)
    
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R

def predict(B0, B1, new_x):
    y = B0 + B1 * new_x
    return y    

# Applying functions to our data & print out the results:
B0, B1, reg_line = linear_regression(x, y)
print('Regression Line: ', reg_line)
R = corr_coef(x, y)
print('Correlation Coef. (i.e. R): ', R)
print('Goodness of Fit (i.e. R^2): ', R**2)

# Prediction based on regression analysis
x_day_stay = 4
print('4-Day Stay Prediction: ', predict(B0, B1, x_day_stay))

fig, ax = plt.subplots(figsize=(8, 4))
text = '''X Mean: {} Years
Y Mean: ${}
R: {}
R^2: {}
y = {} + {}X'''.format(round(x.mean(), 2), 
                       round(y.mean(), 2), 
                       round(R, 4), 
                       round(R**2, 4),
                       round(B0, 3),
                       round(B1, 3))
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='grey', alpha=0.2, pad=1)
# place a text box in upper left in axes coords
ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)
ax.scatter(x, y, alpha=0.75, s=30, linewidths=1, edgecolor='black',c='green')
fig.suptitle('How Hospital Length Affects Charges')
plt.xlabel('Average Length of Stay (days)', fontsize=15)
plt.ylabel('Average Charge', fontsize=15)
plt.plot(x, B0 + B1*x, c='r', linewidth=1, alpha=alpha) # solid_capstyle='round')
ax.grid(True)
# fig.savefig('hospital.png', dpi=300)


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
ax.plot(x_pred, y_pred, '-', color='blue', linewidth=2)
# fig.savefig('propellantfit.png', dpi=300)

#More calculatsions, this time for error
SSyy=sum(np.multiply(y,y))-ysum**2/n
SSE=SSyy-beta1*SSxy
MSE=SSE/(n-2)
s=MSE**(1/2)

r = SSxy/(SSxx*SSyy)**(1/2)
r2 = r**2
r2alt = 1-SSE/SSyy

tstatr = r*(np.sqrt(n-2))/np.sqrt(1-r**2)
pval = stats.t.sf(tstatr, n-2)
print('tstat = ', tstatr)
print('pval = ', pval)

#Confidence interval around Beta and t-test for Beta
tstat=beta1/(s/(SSxx**(1/2)))
tcrit=stats.t.ppf(1-alpha/2,n-2)
UL=beta1+tcrit*s/(SSxx**(1/2))
LL=beta1-tcrit*s/(SSxx**(1/2))

print('We are', 100*(1-alpha), 
'%confident that the true value of the slope is in the interval(', round(LL,2),',',round(UL,2),')')

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
upperp = y_pred + abs(stats.t.ppf(1-alpha/2,n-2)*se_pred(x_pred))
lowerp = y_pred - abs(stats.t.ppf(1-alpha/2,n-2)*se_pred(x_pred))
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


'''Problem 2'''