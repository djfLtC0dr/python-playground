import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.tsa.api as smt
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from patsy import dmatrices
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Read the data 
data = pd.read_csv("./DASC512/student_data.csv", sep = ',')
# print(data)

# explore data.
# data.info()

# subset trng data to only data with Y values => first 456 records
trng_data = data[:456]
# subset_data.info()

# subset our test data all columns except 'Census Tract' and 'Y'
test_data = data.loc[456:506, ~data.columns.isin(['Census Tract', 'Y'])]
# print(test_data)

# create the training data
x_columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X1*X5','X4*X5','X5*X6','X5*X8']
x = trng_data[x_columns]
# fig = sns.pairplot(y)
# skewed right Y so transform
trng_data['tY'], boxlambda = stats.boxcox(trng_data['Y'])
# print(trng_data['tY'])
ty = pd.DataFrame(trng_data['tY'])
# fig = sns.pairplot(ty)

trng_data = x.assign(tY=ty['tY']) 
y_column = ['tY']
y = trng_data[y_column]

## creating function to get model statistics
def get_model_stats():
    x = trng_data[x_columns]
    results = sm.OLS(y, x).fit()
    print(results.summary2())
    return results

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
    fig.show()
    # fig.savefig('Linearity Test.png', dpi=300)

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
    # fig.savefig('Homoscedasticity Test.png', dpi=300)
    fig.show()

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
    # fig.savefig('Final_QQPlot.png', dpi=300)
    fig.show()

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

# remove features w/ variance < 30% => features which mostly remain at the same level 
# across different observations, should not ideally be responsible for differing responses in the observations.   
var = VarianceThreshold(threshold=0.3)
var = var.fit(x,y)
cols = var.get_support(indices=True)
features = x.columns[cols]
# print(features)
x = trng_data[features]
# print(x)

# re-assign our df w/ only the features w/ variance > 30% + our target variable
trng_data = x.assign(tY=y['tY']) 
# print(subset_data)

# Remove features which are not correlated with the response variable 
# plt.figure(figsize=(12,12))
cor = trng_data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

# Consider correlations only with the target variable
cor_target = abs(cor['tY'])

#Select correlations with a correlation above a threshold 10%.
features = cor_target[cor_target>0.1]
# print(features.index)
x_columns = ['X1', 'X2', 'X3', 'X5', 'X6', 'X7', 'X9', 'X10', 'X11', 'X12', 'X1*X5',
            'X4*X5', 'X5*X8']

# Figure out the multicollinearity features to remove via VIF
def compute_vif(considered_features):
    X = trng_data[considered_features]
    # the calculation of variance inflation requires a constant
    X = X.assign(intercept=1)
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif['feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['feature']!='intercept']
    return vif

# VIF dataframe
# vif_data = compute_vif(x_columns)
# print(vif_data)
# compute vif values after removing a feature(s) w/ VIF > 5
x_columns.remove('X1') # VIF 422.005409
# vif_data = compute_vif(x_columns)
# print(vif_data)
x_columns.remove('X10') # VIF 6.349784
vif_data = compute_vif(x_columns)
# print(vif_data)

# Deal w/ outliers
x = trng_data[x_columns]
trng_data = x.assign(tY=y['tY']) 
# trng_data.describe()
Q1 = trng_data.quantile(0.25)
Q3 = trng_data.quantile(0.75)
IQR = Q3 - Q1
index = trng_data[~((trng_data < (Q1 - 1.5 * IQR)) | (trng_data > (Q3 + 1.5 * IQR))).any(axis=1)].index
trng_data.drop(index, inplace=True)
# trng_data.describe()
# trng_data.info()
x = trng_data[x_columns]
y = trng_data['tY']

# fig.savefig('FinalPairPlot_x.png', dpi=300)


# remove the least statistically significant features(s) i.e. pval > 0.05
x_columns.remove('X12') # pval 0.9639 
x_columns.remove('X3') # pval 0.5286
# get_model_stats()
x_columns.remove('X4*X5') # pval 0.1042 
# x_columns.remove('X5') # pval 0.8135
x_columns.remove('X2') # pval 0.2353
x_columns.remove('X9') # pval 0.6558
x_columns.remove('X7') # pval 0.0232
# get_model_stats()
# x_columns.remove('X3') # pval 0.0570

# fig = sns.pairplot(x)

#TODO Skewdness ??
# trng_data['X11'].hist()
x = trng_data[x_columns]
y = trng_data['tY']

model_median_value = sm.OLS(y, x).fit()

y, X = dmatrices('tY ~ X5+X6+X11+Q("X1*X5")+Q("X5*X8")', data = trng_data, return_type ='dataframe')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 50, random_state = 42)
# med_val_test = X_test
# med_val_test['Y'] = y_test
print(model_median_value.summary2(alpha=0.05))
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)

lin_reg = sm.OLS(y,X).fit()
print(lin_reg.summary())

linearity_test(lin_reg, y)
homoscedasticity_test(lin_reg)
normality_of_residuals_test(lin_reg)

acf = smt.graphics.plot_acf(lin_reg.resid, lags=40 , alpha=0.05)

#Durbin Watson Test, the test statistic is between 0 and 4, <2 is positive correlation, >2 is negative correlation
# As a rule of thumb, anything between 1.5 and 2.5 is OK
DW = sms.durbin_watson(lin_reg.resid)
print(DW)
print(lin_reg.summary2())
# print(X_test)
# print(y_test)
calculate_rmse(lin_reg, X_test, y_test)

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

# # Determine Regression 
# ols = LinearRegression()
# X = trng_data[x_columns]
# model = ols.fit(X, y)
# # print(model.coef_)
# # print(model.intercept_)
# # print(model.score(X, y))
# A = round(model.intercept_[0], 4)
# # print(A)
# X6 = round(model.coef_[0][0], 4)
# X11 = round(model.coef_[0][1], 4)
# X12 = round(model.coef_[0][2], 4)
# X1_X5 = round(model.coef_[0][3], 4)
# X4_X5 = round(model.coef_[0][4], 4)
# X5_X8 = round(model.coef_[0][5], 4)

# median_value = A + X6 + X11 + X12 + X1_X5 + X4_X5 + X5_X8
# print('median_value = ' + str(A) + ' + ' + str(X11) + ' + ' + str(X12) + 
#     ' + ' + str(X1_X5) + ' + ' + str(X4_X5) + ' + ' + str(X5_X8) +' => ' , median_value)

# # Plot our model using Test/Train Data
# fig, ax = plt.subplots(figsize=(10,6))
# fig.suptitle('Median Value Owner-Occupied Homes Test/Train Prediction Plot')
# fig.tight_layout(pad=3)
# x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=50,random_state=42)
# linreg=LinearRegression()
# linreg.fit(x_train,y_train)
# #test_data = test_data = data[456:506] ['X6', 'X11', 'X12', 'X1*X5', 'X4*X5', 'X5*X8']
# arr_y_pred=linreg.predict(test_data) 
# y_pred_df = pd.DataFrame(arr_y_pred)
# y_pred_df.columns = ['y_pred']
# # print(y_pred_df['y_pred'])
# sns.regplot(x=y_test,y=y_pred_df,ci=95,marker='o',color ='blue')
# ax.grid()

# #calculate prediction intervals
# prediction=model_median_value.get_prediction(test_data)
# predints=prediction.summary_frame(alpha=0.05)
# # print(predints)

# ax.fill_between(y_pred_df['y_pred'], predints['mean_ci_lower'], predints['mean_ci_upper'], color='#888888', alpha=0.6)
# ax.fill_between(y_pred_df['y_pred'], predints['obs_ci_lower'], predints['obs_ci_upper'], color='#888888', alpha=0.2)
# fig.savefig('median_value_test_train_predict_plot.png', dpi=300)
