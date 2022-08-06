import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
# clean-up unnecessary data
df_crime.drop(df_crime.columns[df_crime.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
print(df_crime.head())


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
