from statistics import mode
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

# *********Regression Functions************
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

def calc_xy_stats(x, y):
    x_mean=np.mean(x)
    x_sum=sum(x)
    y_mean=np.mean(y)
    y_sum=sum(y)
    xy=np.multiply(x,y)
    xx=np.multiply(x,x)
    yy=np.multiply(y,y)
    sum_xy=sum(xy)
    sum_xx=sum(xx)
    n=len(x)  
    return (x_mean, x_sum, y_mean, y_sum, xy, xx, yy, sum_xy, sum_xx, n)

def calc_betas_fitted(sum_xy, x_sum, y_sum, n, sum_xx, y_mean, x_mean):
    SSxy=sum_xy-x_sum*y_sum/n
    SSxx=sum_xx-x_sum**2/n
    beta_1=SSxy/SSxx
    beta_0=y_mean-beta_1*x_mean
    fitted = lambda xx: beta_0 + beta_1*xx  
    return (SSxy, SSxx, beta_1, beta_0, fitted)

def calc_error(yy, y_sum, n, beta_1, SSxy):
    SSyy=sum(yy)-y_sum**2/n
    SSE=SSyy-beta_1*SSxy
    MSE=SSE/(n-2)
    s=MSE**(1/2)
    return (SSyy, SSE, MSE, s)

def calc_coeffs(SSxy, SSxx, SSyy, SSE):
    r = SSxy/(SSxx*SSyy)**(1/2)
    r2 = r**2
    r2_alt = 1-SSE/SSyy
    return (r, r2, r2_alt)

def corr_coef(x, y):
    N = len(x)
    
    num = (N * (x*y).sum()) - (x.sum() * y.sum())
    den = np.sqrt((N * (x**2).sum() - x.sum()**2) * (N * (y**2).sum() - y.sum()**2))
    R = num / den
    return R    

def calc_tstats(r, n):
    tstat_r = r*(np.sqrt(n-2))/np.sqrt(1-r**2)
    pval = stats.t.sf(tstat_r, n-2)
    return (tstat_r, pval)

def predict(B0, B1, new_x):
    y = B0 + B1 * new_x
    return y    

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

# Calculate xy stats
x_mean, x_sum, y_mean, y_sum, xy, xx, yy, sum_xy, sum_xx, n = calc_xy_stats(x, y)

# Applying functions to our data & print out the results:
B0, B1, reg_line = linear_regression(x, y)
print('Regression Line: ', reg_line)
R = corr_coef(x, y)
print('Correlation Coef. (i.e. R): ', R)
print('Goodness of Fit (i.e. R^2): ', R**2)

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
plt.plot(x, B0 + B1*x, c='r', linewidth=1, alpha=alpha)
ax.grid(True)
# fig.savefig('hospital.png', dpi=300)

# Calculate betas and fitted line
SSxy, SSxx, beta_1, beta_0, fitted = calc_betas_fitted(sum_xy, x_sum, y_sum, n, sum_xx, y_mean, x_mean)

# Create a plot with the fitted line
x_pred = np.linspace(x.min(), x.max(), 50)
y_pred=fitted(x_pred)
ax.plot(x_pred, y_pred, '-', color='blue', linewidth=2)
# fig.savefig('propellantfit.png', dpi=300)

#More calculatsions, this time for error
SSyy, SSE, MSE, s = calc_error(yy, y_sum, n, beta_1, SSxy)

# Calc our coefficients
r, r2, r2_alt = calc_coeffs(SSxy, SSxx, SSyy, SSE)

# Calc our tstats
tstat_r, pval = calc_tstats(r, n)
print('tstat = ', tstat_r)
print('pval = ', pval)

#Confidence interval around Beta and t-test for Beta
tstat=beta_1/(s/(SSxx**(1/2)))
tcrit=stats.t.ppf(1-alpha/2,n-2)
UL=beta_1+tcrit*s/(SSxx**(1/2))
LL=beta_1-tcrit*s/(SSxx**(1/2))

print('We are', 100*(1-alpha), 
'%confident that the true value of the slope is in the interval(', round(LL,2),',',round(UL,2),')')

# Prediction based on regression analysis
x_day_stay = 4
print('4-Day Stay Prediction: ', predict(B0, B1, x_day_stay))

#Estimation and Prediction Interval calculations
se_est = lambda x: s * np.sqrt(  1./n + (x-x_mean)**2/SSxx)
se_pred = lambda x: s * np.sqrt(1+1./n + (x-x_mean)**2/SSxx)

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
df_len_chrg=pd.DataFrame({'length':x,'charge':y})
model_hospital=smf.ols('charge~length',df_len_chrg).fit()
print(model_hospital.summary2(alpha=alpha))

# anova_table = sm.stats.anova_lm(model_hospital)
# anova_table[:-1]['sum_sq']/sum(anova_table['sum_sq'])

'''Problem 2 Using the ‘hofbatting.csv’ 
Perform a regression analysis to determine if OBP can be used to predict SLG.'''
#Read the data and assign 'x' and 'y'
df_hof = pd.read_csv("hofbatting.csv", sep = ',')
# print(df_hof.head())
# rename the unnamed column to something useful
df_hof.rename(columns = {'Unnamed: 1':'Inductee'}, inplace = True)
# print(df_hof.head())
x = df_hof['OBP']
y = df_hof['SLG']
alpha = 0.05

# Calculate xy stats
x_mean, x_sum, y_mean, y_sum, xy, xx, yy, sum_xy, sum_xx, n = calc_xy_stats(x, y)

# Applying functions to our data & print out the results:
B0, B1, reg_line = linear_regression(x, y)
print('Regression Line: ', reg_line)
R = corr_coef(x, y)
print('Correlation Coef. (i.e. R): ', R)
print('Goodness of Fit (i.e. R^2): ', R**2)

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
fig.suptitle('How OBP Affects SLG')
plt.xlabel('On-base Pct (OBP)', fontsize=15)
plt.ylabel('Slugging Pct (SLG)', fontsize=15)
plt.plot(x, B0 + B1*x, c='r', linewidth=1, alpha=alpha)
ax.grid(True)
# fig.savefig('hospital.png', dpi=300)

# Calculate betas and fitted line
SSxy, SSxx, beta_1, beta_0, fitted = calc_betas_fitted(sum_xy, x_sum, y_sum, n, sum_xx, y_mean, x_mean)

# Create a plot with the fitted line
x_pred = np.linspace(x.min(), x.max(), 50)
y_pred=fitted(x_pred)
ax.plot(x_pred, y_pred, '-', color='blue', linewidth=2)
# fig.savefig('propellantfit.png', dpi=300)

#More calculatsions, this time for error
SSyy, SSE, MSE, s = calc_error(yy, y_sum, n, beta_1, SSxy)

# Calc our coefficients
r, r2, r2_alt = calc_coeffs(SSxy, SSxx, SSyy, SSE)

# Calc our tstats
tstat_r, pval = calc_tstats(r, n)
print('tstat = ', tstat_r)
print('pval = ', pval)

#Confidence interval around Beta and t-test for Beta
tstat=beta_1/(s/(SSxx**(1/2)))
tcrit=stats.t.ppf(1-alpha/2,n-2)
UL=beta_1+tcrit*s/(SSxx**(1/2))
LL=beta_1-tcrit*s/(SSxx**(1/2))

print('We are', 100*(1-alpha), 
'%confident that the true value of the slope is in the interval(', round(LL,2),',',round(UL,2),')')

# Prediction based on regression analysis
x_obp = 0.40
print('SLG Pct based on 0.40 OBP: ', predict(B0, B1, x_obp))

#output Model Summary
df_obp_slg=pd.DataFrame({'obp':x,'slg':y})
model_baseball=smf.ols('slg~obp',df_obp_slg).fit()
print(model_baseball.summary2(alpha=alpha))
beta_0_with_outlier = model_baseball.params[0]
beta_1_with_outlier = model_baseball.params[1]
fitted = lambda xx: beta_0_with_outlier + beta_1_with_outlier*xx

# Create a plot with the fitted line
x_pred = np.linspace(x.min(), x.max(), 50)
y_pred=fitted(x_pred)
ax.plot(x_pred, y_pred, '-', color='blue', linewidth=2)
# fig.savefig('propellantfit.png', dpi=300)

#calculate prediction intervals
df_pred = pd.DataFrame({'obp': x_pred,'slg': y_pred})
prediction=model_baseball.get_prediction(df_pred)
predints=prediction.summary_frame(alpha=0.1)

ax.fill_between(x_pred, predints['mean_ci_lower'], predints['mean_ci_upper'], color='#888888', alpha=0.6)
ax.fill_between(x_pred, predints['obs_ci_lower'], predints['obs_ci_upper'], color='#888888', alpha=0.2)

# Examine Regression w/o Willard Brown outlier => index 20
# print(df_hof.head(25))
# print(df_obp_slg.head(25))
# print(df_obp_slg.loc[20])
df_obp_slg = df_obp_slg.drop(index=20)

x = df_obp_slg['obp']
y = df_obp_slg['slg']
alpha = 0.05

# Calculate xy stats
x_mean, x_sum, y_mean, y_sum, xy, xx, yy, sum_xy, sum_xx, n = calc_xy_stats(x, y)

# Applying functions to our data & print out the results:
B0, B1, reg_line = linear_regression(x, y)
print('Regression Line: ', reg_line)
R = corr_coef(x, y)
print('Correlation Coef. (i.e. R): ', R)
print('Goodness of Fit (i.e. R^2): ', R**2)

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
fig.suptitle('How OBP Affects SLG--without Outlier')
plt.xlabel('On-base Pct (OBP)', fontsize=15)
plt.ylabel('Slugging Pct (SLG)', fontsize=15)
plt.plot(x, B0 + B1*x, c='r', linewidth=1, alpha=alpha)
ax.grid(True)
# fig.savefig('hospital.png', dpi=300)

# Calculate betas and fitted line
SSxy, SSxx, beta_1, beta_0, fitted = calc_betas_fitted(sum_xy, x_sum, y_sum, n, sum_xx, y_mean, x_mean)

# Create a plot with the fitted line
x_pred = np.linspace(x.min(), x.max(), 50)
y_pred=fitted(x_pred)
ax.plot(x_pred, y_pred, '-', color='blue', linewidth=2)
# fig.savefig('propellantfit.png', dpi=300)

#More calculatsions, this time for error
SSyy, SSE, MSE, s = calc_error(yy, y_sum, n, beta_1, SSxy)

# Calc our coefficients
r, r2, r2_alt = calc_coeffs(SSxy, SSxx, SSyy, SSE)

# Calc our tstats
tstat_r, pval = calc_tstats(r, n)
print('tstat = ', tstat_r)
print('pval = ', pval)

#Confidence interval around Beta and t-test for Beta
tstat=beta_1/(s/(SSxx**(1/2)))
tcrit=stats.t.ppf(1-alpha/2,n-2)
UL=beta_1+tcrit*s/(SSxx**(1/2))
LL=beta_1-tcrit*s/(SSxx**(1/2))

print('We are', 100*(1-alpha), 
'%confident that the true value of the slope is in the interval(', round(LL,2),',',round(UL,2),')')

# Prediction based on regression analysis
x_obp = 0.40
print('SLG Pct based on 0.40 OBP: ', predict(B0, B1, x_obp))

#output Model Summary
df_obp_slg=pd.DataFrame({'obp':x,'slg':y})
model_baseball=smf.ols('slg~obp',df_obp_slg).fit()
print(model_baseball.summary2(alpha=alpha))
beta_0_without_outlier = model_baseball.params[0]
beta_1_without_outlier = model_baseball.params[1]
fitted = lambda xx: beta_0_without_outlier + beta_1_without_outlier*xx

# Create a plot with the fitted line
x_pred = np.linspace(x.min(), x.max(), 50)
y_pred=fitted(x_pred)
ax.plot(x_pred, y_pred, '-', color='blue', linewidth=2)
# fig.savefig('propellantfit.png', dpi=300)

#calculate prediction intervals
df_pred = pd.DataFrame({'obp': x_pred,'slg': y_pred})
prediction=model_baseball.get_prediction(df_pred)
predints=prediction.summary_frame(alpha=0.1)

ax.fill_between(x_pred, predints['mean_ci_lower'], predints['mean_ci_upper'], color='#888888', alpha=0.6)
ax.fill_between(x_pred, predints['obs_ci_lower'], predints['obs_ci_upper'], color='#888888', alpha=0.2)

print('Difference in slope b/t w/ and w/o outliers is ' + str(1 - (beta_1_with_outlier/beta_1_without_outlier)))
