import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
#Read the data 
df_team = pd.read_csv("teamdata.csv", sep = ',')
# print(df_team.head())
# clean-up unnecessary data
df_team.drop(df_team.columns[df_team.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
# print(df_team.head())
# setup our dataframe for x,y data
df_team['run_diff'] = df_team['R'] - df_team['RA'] # R−RA
df_team['win_pct'] = df_team['W'] / (df_team['W'] + df_team['L']) # W/(W + L)
# print(df_team.head())
x = df_team['run_diff']
y = df_team['win_pct']
alpha = 0.05

# Calculate xy stats
x_mean, x_sum, y_mean, y_sum, xy, xx, yy, sum_xy, sum_xx, n = calc_xy_stats(x, y)

# Calculate betas and fitted line
SSxy, SSxx, beta_1, beta_0, fitted = calc_betas_fitted(sum_xy, x_sum, y_sum, n, sum_xx, y_mean, x_mean)

#More calculatsions, this time for error
SSyy, SSE, MSE, s = calc_error(yy, y_sum, n, beta_1, SSxy)

# Calc our coefficients
r, r2, r2_alt = calc_coeffs(SSxy, SSxx, SSyy, SSE)

# Calc our tstats
tstat_r, pval = calc_tstats(r, n)
print('tstat = ', tstat_r)
print('pval = ', pval)

#output Model Summary
model_baseball=smf.ols('win_pct~run_diff', df_team).fit()
print(model_baseball.summary2(alpha=alpha))

#Pull residuals
residuals = model_baseball.resid
fitted_values = model_baseball.fittedvalues

#plot Residuals vs Fitted Values

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(fitted_values, residuals, alpha=1.0, color='dodgerblue')
fig.suptitle('Residuals versus Fitted Values')
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
fig.tight_layout(pad=3)
ax.grid(True)
plt.axvline(x=min(fitted_values)+(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axvline(x=min(fitted_values)+2*(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axhline(y=0,color='black')
# fig.savefig('baseball_residuals_run_diff.png', dpi=300)

#***********************************************************************
# Examine Regression w/ Pythagorean Expectation => R**2 / (R**2 + RA**2)
#***********************************************************************
df_team['py_exp'] = df_team['R']**2 / (df_team['R']**2 + df_team['RA']**2)
# drop the run_diff data
df_team.drop(df_team.columns['run_diff'],axis = 1, inplace = True)
# print(df_team.head())
x = df_team['py_exp']
y = df_team['win_pct']
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
ax.scatter(x, y, alpha=0.75, s=0.5, linewidths=1, edgecolor='black',c='green')
fig.suptitle('How Pythagorean Expectation Affects Winning Pct')
plt.xlabel('Pythagorean Exp', fontsize=15)
plt.ylabel('Winning Pct', fontsize=15)
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

#output Model Summary
model_baseball=smf.ols('win_pct~py_exp', df_team).fit()
print(model_baseball.summary2(alpha=alpha))

#Pull residuals
residuals = model_baseball.resid
fitted_values = model_baseball.fittedvalues

#plot Residuals vs Fitted Values

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(fitted_values, residuals, alpha=1.0, color='dodgerblue')
fig.suptitle('Residuals versus Fitted Values')
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
fig.tight_layout(pad=3)
ax.grid(True)
plt.axvline(x=min(fitted_values)+(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axvline(x=min(fitted_values)+2*(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axhline(y=0,color='black')
# fig.savefig('baseball_residuals_run_diff.png', dpi=300)
