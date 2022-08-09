import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor

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

#output Model Summary
model_baseball_run_diff=smf.ols('win_pct~run_diff', df_team).fit()
print(model_baseball_run_diff.summary2(alpha=alpha))

#Pull residuals
residuals = model_baseball_run_diff.resid
fitted_values = model_baseball_run_diff.fittedvalues

#plot Residuals vs Fitted Values

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(fitted_values, residuals, alpha=1.0, color='dodgerblue')
fig.suptitle('Residuals versus Fitted Values - Run Diff')
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
fig.tight_layout(pad=3)
ax.grid(True)
plt.axvline(x=min(fitted_values)+(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axvline(x=min(fitted_values)+2*(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axhline(y=0,color='black')
# fig.savefig('baseball_residuals_run_diff.png', dpi=300)

#***********************************************************************
# Examine Residuals w/ Pythagorean Expectation => R**2 / (R**2 + RA**2)
#***********************************************************************
df_team['py_exp'] = df_team['R']**2 / (df_team['R']**2 + df_team['RA']**2)
# drop the run_diff data
df_team.drop('run_diff', axis=1, inplace=True)
# print(df_team.head())
x = df_team['py_exp']
y = df_team['win_pct']
alpha = 0.05

#output Model Summary
model_baseball_py_exp=smf.ols('win_pct~py_exp', df_team).fit()
print(model_baseball_py_exp.summary2(alpha=alpha))

#Pull residuals
residuals = model_baseball_py_exp.resid
fitted_values = model_baseball_py_exp.fittedvalues

#plot Residuals vs Fitted Values

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(fitted_values, residuals, alpha=1.0, color='dodgerblue')
fig.suptitle('Residuals versus Fitted Values - Pythagorean Exp')
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
fig.tight_layout(pad=3)
ax.grid(True)
plt.axvline(x=min(fitted_values)+(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axvline(x=min(fitted_values)+2*(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axhline(y=0,color='black')
# fig.savefig('baseball_residuals_py_exp.png', dpi=300)

'''Problem 2'''
# Using the data file ‘UScrime.csv’ fit a model that can be used to predict 
# the rate of offenses per 1000000 population in 1960 (Achieve an R a 2 ≥ 0.7). 
# Run residual analysis (graphically) to determine if your model is accurate.

#Read the data 
df_crime = pd.read_csv("UScrime.csv", sep = ',')
# print(df_crime.head())
# checking missing values in the data
df_crime.isnull().sum()
# creating the training data
y_column = ['Crime']
y = df_crime[y_column]
x_columns = ['M', 'So', 'Ed', 'Po1', 'Po2', 'LF', 'M.F', 'Pop', 'NW', 'U1', 'U2', 'Wealth', 'Ineq', 'Prob', 'Time']
x = df_crime[x_columns]

# remove features w/ variance < 30% => features which mostly remain at the same level 
# across different observations, should not ideally be responsible for differing responses in the observations.   
var = VarianceThreshold(threshold=0.3)
var = var.fit(x,y)
cols = var.get_support(indices=True)
features = x.columns[cols]
# print(features)
x = df_crime[features]
# print(x)

# re-assign our df w/ only the features w/ variance > 30% + our target variable
df_crime = x.assign(crime=y['Crime']) 
# print(df_crime)

# Remove features which are not correlated with the response variable 
plt.figure(figsize=(12,12))
cor = df_crime.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Consider correlations only with the target variable
cor_target = abs(cor['crime'])

#Select correlations with a correlation above a threshold 10%.
features = cor_target[cor_target>0.1]
print(features.index)
x_columns = ['Ed', 'Po1', 'Po2', 'M.F', 'Pop', 'U2', 'Wealth', 'Ineq', 'Time']

# Figure out the multicollinearity features to remove via VIF
def compute_vif(considered_features):
    
    X = df_crime[considered_features]
    # the calculation of variance inflation requires a constant
    X = X.assign(intercept=1)
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif['feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['feature']!='intercept']
    return vif

# VIF dataframe
vif_data = compute_vif(x_columns)
# print(vif_data)
# compute vif values after removing a feature(s) w/ VIF > 5
x_columns.remove('Po2') # VIF 94.093117
vif_data = compute_vif(x_columns)
# print(vif_data)
x_columns.remove('Wealth') # VIF 8.841141
vif_data = compute_vif(x_columns)
print(vif_data)

## creating function to get model statistics
def get_stats():
    x = df_crime[x_columns]
    results = sm.OLS(y, x).fit()
    print(results.summary2())
get_stats()

# remove the least statistically significant variable(s)
x_columns.remove('Time') # pval 0.5860
get_stats()
x_columns.remove('Pop') # pval 0.1135
get_stats()
x_columns.remove('Ed') # pval 0.1178 
get_stats()

# TODO Run residual analysis (graphically) to determine if model is accurate.

#*********************************************************
from matplotlib.backends.backend_pdf import PdfPages
# Save all figures to PDF
def save_figs_pdf(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

save_figs_pdf('hw6_plt_figs.pdf')
