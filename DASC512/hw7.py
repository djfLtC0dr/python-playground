import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import numpy as np
import seaborn as sns
import math
from statsmodels.stats import power
from statsmodels.stats.diagnostic import lilliefors
from statsmodels.stats import weightstats as stests
from sklearn import datasets
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import linear_model
from itertools import chain
from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
sns.set_style('darkgrid')
sns.mpl.rcParams['figure.figsize'] = (15.0, 9.0)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.tsa.api as smt

def linearity_test(model, y):
    '''
    Function for visually inspecting the assumption of linearity in a linear regression model.
    It plots observed vs. predicted values and residuals vs. predicted values.
    Args:
    * model - fitted OLS model from statsmodels
    * y - observed values
    '''
    fitted_vals = model.predict()
    resids = model.resid

    fig, ax = plt.subplots(1, 2)

    sns.regplot(x=fitted_vals, y=y, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Observed vs. Predicted Values', fontsize=16)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=16)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    fig.savefig('Linearity Test.png', dpi=300)

def homoscedasticity_test(model):
    '''
    Function for testing the homoscedasticity of residuals in a linear regression model.
    It plots residuals and standardized residuals vs. fitted values and runs Breusch-Pagan and Goldfeld-Quandt tests.
    Args:
    * model - fitted OLS model from statsmodels
    '''
    fitted_vals = model.predict()
    resids = model.resid
    resids_standardized = model.get_influence().resid_studentized_internal

    fig, ax = plt.subplots(1, 2)

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[0], line_kws={'color': 'red'})
    ax[0].set_title('Residuals vs Fitted', fontsize=16)
    ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Scale-Location', fontsize=16)
    ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')
    fig.savefig('Homoscedasticity Test.png', dpi=300)

    bp_test = pd.DataFrame(sms.het_breuschpagan(resids, model.model.exog),
                           columns=['value'],
                           index=['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value'])

    gq_test = pd.DataFrame(sms.het_goldfeldquandt(resids, model.model.exog)[:-1],
                           columns=['value'],
                           index=['F statistic', 'p-value'])

    print('\n Breusch-Pagan test ----')
    print(bp_test)
    print('\n Goldfeld-Quandt test ----')
    print(gq_test)
    print('\n Residuals plots ----')

def normality_of_residuals_test(model):
    '''
    Function for drawing the normal QQ-plot of the residuals and running 4 statistical tests to
    investigate the normality of residuals.
    Arg:
    * model - fitted OLS models from statsmodels
    '''
    fig = sm.ProbPlot(model.resid).qqplot(line='s')
    plt.title('Q-Q plot')
    fig.savefig('QQ Plot.png', dpi=300)

    jb = stats.jarque_bera(model.resid)
    sw = stats.shapiro(model.resid)
    ad = stats.anderson(model.resid, dist='norm')
    ks = stats.kstest(model.resid, 'norm')

    print(f'Jarque-Bera test ---- statistic: {jb[0]:.4f}, p-value: {jb[1]}')
    print(f'Shapiro-Wilk test ---- statistic: {sw[0]:.4f}, p-value: {sw[1]:.4f}')
    print(f'Kolmogorov-Smirnov test ---- statistic: {ks.statistic:.4f}, p-value: {ks.pvalue:.4f}')
    print(f'Anderson-Darling test ---- statistic: {ad.statistic:.4f}, 5% critical value: {ad.critical_values[2]:.4f}')
    print('If the returned AD statistic is larger than the critical value, then for the 5% significance level, the null hypothesis that the data come from the Normal distribution should be rejected. ')

def calculate_rmse(model, X_test, y_test):
    '''
    Function for calculating the rmse of a test set
    Parameters
    ----------
    model : OLS model from Statsmodels ( formula or regular )
    X_test : Test inputs ( MUST MATCH MODEL INPUTS )
    y_test : Test outputs ( MUST INCLUDE ANY TRANFORMATIONS TO MODEL OUTPUTS )
    Returns
    -------
    Returns RMSE printed in the console
    '''
    predicted = model.predict(X_test)
    MSE = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(MSE)
    print(rmse)

diabetes = pd.read_table('diabetes.txt', sep ='\s+')
y, X = dmatrices('Y ~ AGE+C(SEX)+BMI+BP+S1+S2+S3+S4+S5+S6', data = diabetes, return_type ='dataframe')
X.columns = ['Intercept', 'SEX', 'AGE', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
diabetes['tY'], boxlambda = stats.boxcox(diabetes['Y'])
diabetes['log_Y'] = np.log(diabetes['Y'])
diabetes['inv_Y'] = 1/diabetes['Y']
diabetes['Y_sqrd'] = diabetes['Y']**2
Y_mean = np.mean(diabetes['Y'])
diabetes['center_Y'] = diabetes['Y'] - Y_mean
diabetes['log_S4'] = np.log(diabetes['S4'])
diabetes['inv_S4'] = 1/diabetes['S4']
S4_mean = np.mean(diabetes['S4'])
diabetes['center_S4'] = diabetes['S4'] - S4_mean
diabetes['log_AGE'] = np.log(diabetes['AGE'])
BMI_mean = np.mean(diabetes['BMI'])
diabetes['center_BMI'] = diabetes['BMI'] - BMI_mean
diabetes['log_BMI'] = np.log(diabetes['BMI'])
diabetes = diabetes.drop(index=156)
diabetes = diabetes.drop(index=56)
diabetes = diabetes.drop(index=92)
spec = inv_boxcox(diabetes.tY, boxlambda)
model = smf.ols('tY ~ C(SEX)+BMI+BP+S1+S3+S5', diabetes).fit()
y, X = dmatrices('tY ~ C(SEX)+BMI+BP+S1+S3+S5', data = diabetes, return_type ='dataframe')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
diabetes_test = X_test
diabetes_test['Y'] = y_test
print(model.summary2(alpha=0.05))
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

lin_reg = sm.OLS(y,X).fit()
print(lin_reg.summary())

linearity_test(lin_reg, y)
homoscedasticity_test(lin_reg)

acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05)
acf.show()
normality_of_residuals_test(lin_reg)

DW = sms.durbin_watson(lin_reg.resid)
print(DW)

calculate_rmse(lin_reg, X_test.drop(['Y'],axis=1), y_test)

for column in X.columns:
    corr_test = stats.stats.pearsonr(X[column], lin_reg.resid)
    print(f'Variable: {column} --- correlation: {corr_test[0]:.4f}, p-value: {corr_test[1]:.4f}')

###  Build Every Possible Model
from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

mylist = list(powerset(list(X.columns)))
mylist = [list(row) for row in mylist]

##Target is AIC
AIC_scores = pd.DataFrame(columns=["AIC"])
for i in range(len(mylist)):
    AIC_scores.loc[i, 'AIC'] = sm.OLS(y, X[mylist[i]]).fit().aic

print(AIC_scores.sort_values(by='AIC').head())

# original full model
# print(mylist[1557]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S5']
# print(mylist[1874]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S4', 'S5']
# print(mylist[1876]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S5', 'S6']
# print(mylist[1562]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S4', 'S5']
# print(mylist[1560]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S3', 'S5']

# original full model + log_S4
# print(mylist[2638]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S2', 'S5']
# print(mylist[2642]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S3', 'S5']
# print(mylist[3432]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S2', 'S4', 'S5']
# print(mylist[2645]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S4', 'S5']
# print(mylist[2652]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S2', 'S3', 'S5']

# original full model with log_AGE
# print(mylist[1557]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S2', 'S5']
# print(mylist[1560]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S3', 'S5']
# print(mylist[1874]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S2', 'S4', 'S5']
# print(mylist[1562]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S1', 'S4', 'S5']
# print(mylist[1566]) # ['Intercept', 'C(SEX)[T.1.0]', 'BMI', 'BP', 'S2', 'S3', 'S5']

# Full model with boxcox y-values
# print(mylist[1557]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S5']
# print(mylist[1876]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S5', 'S6']
# print(mylist[1874]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S4', 'S5']
# print(mylist[1817]) # ['Intercept', 'C(SEX)[T.2]', 'AGE', 'BMI', 'BP', 'S1', 'S2', 'S5']
# print(mylist[1872]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S5']

# Y ~ C(SEX)+BMI+BP+S1+S4+S5+S6
# print(mylist[246]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S4', 'S5']
# print(mylist[254]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S4', 'S5', 'S6']
# print(mylist[219]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S5']
# print(mylist[248]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S5', 'S6']
# print(mylist[233]) # ['Intercept', 'BMI', 'BP', 'S1', 'S4', 'S5']

# Full model with boxcox y-values dropped 156,56,92 outliers
# print(mylist[1557]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S5']
# print(mylist[1876]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S5', 'S6']
# print(mylist[1872]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S5']
# print(mylist[1874]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S2', 'S4', 'S5']
# print(mylist[1817]) # ['Intercept', 'C(SEX)[T.2]', 'AGE', 'BMI', 'BP', 'S1', 'S2', 'S5']

# # tY ~ C(SEX)+BMI+BP+S1+S3+S5 boxcox y-values dropped 156,56,92 outliers
# print(mylist[125]) # ['C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S3', 'S5']
# print(mylist[126]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S3', 'S5']
# print(mylist[115]) # ['C(SEX)[T.2]', 'BMI', 'BP', 'S3', 'S5']
# print(mylist[121]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S3', 'S5']
# print(mylist[120]) # ['Intercept', 'C(SEX)[T.2]', 'BMI', 'BP', 'S1', 'S5']

# fig = sns.pairplot(X)
# fig.savefig('Hw7 Pair Plot3.png', dpi=300)

# plt.show()