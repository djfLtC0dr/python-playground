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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  RandomForestRegressor
from sklearn.pipeline import Pipeline
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")
    
def get_model_stats(x, y):
    # x = trng_data[x_columns]
    results = sm.OLS(y, x).fit()
    print(results.summary2())
    return results


def compute_vif(considered_features):
    '''
    Function to list out the multicollinearity features to remove via VIF >5
    '''
    X = trng_data[considered_features]
    # the calculation of variance inflation requires a constant
    X = X.assign(intercept=1)
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif['feature'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['feature']!='intercept']
    return vif    

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
    ax[0].set_title('Observed vs. Predicted Values', fontsize=10)
    ax[0].set(xlabel='Predicted', ylabel='Observed')

    sns.regplot(x=fitted_vals, y=resids, lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Residuals vs. Predicted Values', fontsize=10)
    ax[1].set(xlabel='Predicted', ylabel='Residuals')
    fig.show()
    fig.savefig('finalLinearityTest.png', dpi=300)

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
    ax[0].set_title('Residuals vs Fitted', fontsize=10)
    ax[0].set(xlabel='Fitted Values', ylabel='Residuals')

    sns.regplot(x=fitted_vals, y=np.sqrt(np.abs(resids_standardized)), lowess=True, ax=ax[1], line_kws={'color': 'red'})
    ax[1].set_title('Scale-Location', fontsize=10)
    ax[1].set(xlabel='Fitted Values', ylabel='sqrt(abs(Residuals))')
    fig.savefig('finalHomoscedasticityTest.png', dpi=300)
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
    fig.savefig('finalQQPlot.png', dpi=300)
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

# Read the data 
data = pd.read_csv("student_data.csv", sep = ',')
# print(data)
data.rename({'X1*X5': 'X1_X5', 'X4*X5': 'X4_X5', 'X5*X6': 'X5_X6', 'X5*X8': 'X5_X8'}, axis=1, inplace=True)
# print(data)
# explore data.
# data.info()

# Fill the missing values of the 'Y' column with the median of the non-missing values
data['Y'].fillna(data['Y'].median(),inplace=True)

# subset trng data to only data with non-missing Y values => first 456 records
trng_data = pd.DataFrame(data.loc[:456, ~data.columns.isin(['Census Tract'])])
# print(trng_data)
# subset_data.info()

# Drop outliers
Q1 = trng_data.quantile(0.25)
Q3 = trng_data.quantile(0.75)
IQR = Q3 - Q1
index = trng_data[~((trng_data < (Q1 - 1.5 * IQR)) | (trng_data > (Q3 + 1.5 * IQR))).any(axis=1)].index
trng_data.drop(index, inplace=True)
# trng_data.describe()
# trng_data.info()

# separate the independent and target variable 
X = trng_data.drop(columns=['Y'])
y = trng_data['Y']
X=sm.add_constant(X)
print(X.columns)

# create the training data
x_columns = ['const', 'X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X1_X5','X4_X5','X5_X6','X5_X8']
# x = trng_data[x_columns]
y_column = ['Y']
# y = trng_data[y_column]

###  Build Every Possible Model
from itertools import chain, combinations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

mylist = list(powerset(list(X.columns)))
mylist = [list(row) for row in mylist]

##Target is AIC
# AIC_scores = pd.DataFrame(columns=["AIC"])
# for i in range(len(mylist)):
#     AIC_scores.loc[i, 'AIC'] = sm.OLS(y, X[mylist[i]]).fit().aic

# print(AIC_scores.sort_values(by='AIC').head())

#               AIC
# 33988  4327.624654
# 81959  4327.692603
# 57506   4327.81554
# 88025   4327.92411
# 54866  4327.950256
print(mylist[33988]) # => ['X1', 'X5', 'X6', 'X8', 'X11', 'X12', 'X5_X6']
print(mylist[81959]) # => ['X1', 'X3', 'X4', 'X5', 'X6', 'X8', 'X11', 'X12', 'X5_X6']
print(mylist[57506]) # => ['X1', 'X4', 'X5', 'X6', 'X8', 'X11', 'X12', 'X5_X6']
print(mylist[88025]) # => ['X3', 'X4', 'X5', 'X6', 'X8', 'X11', 'X12', 'X1_X5', 'X5_X6']
print(mylist[54866]) # => ['X1', 'X2', 'X5', 'X6', 'X8', 'X11', 'X12', 'X5_X6']

###  Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.ensemble import AdaBoostRegressor
estimator = AdaBoostRegressor(random_state=42, n_estimators=100)
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
filter = selector.support_
print("Num Features: %d" % selector.n_features_)
print("Selected Features: %s" % filter)
print("Feature Ranking: %s" % selector.ranking_)
features = X.columns
print("All features:")
print(features)

print("Selected features:")
print(features[filter]) 
# => 'X6', 'X8', 'X10', 'X11', 'X12', 'X1_X5', 'X5_X6', 'X5_X8'
# x_columns = ['X5', 'X6', 'X11', 'X1_X5', 'X5_X8']


# step-wise regression by P>|t|
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
lr = LinearRegression()

# Build step forward feature selection
sfs1 = sfs(lr, 
          k_features=7, 
          forward=True, 
          floating=False, 
          scoring='neg_mean_squared_error',
          fixed_features=(13, 14, 15, 16), # Final rqmmts 'X1_X5','X4_X5','X5_X6','X5_X8' 
          cv=10)

# Perform SFFS
feature_names = tuple(x_columns)
sfs1 = sfs1.fit(X, y, custom_feature_names=feature_names)
fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
print(sfs1.k_feature_names_)
print(sfs1.k_score_)

impt_features = list(sfs1.k_feature_names_) # ['X6', 'X12', 'X1_X5', 'X4_X5', 'X5_X6', 'X5_X8']

X = trng_data[impt_features]
# VIF dataframe
# x_columns.remove('X1') #503.224622
# x_columns.remove('X3') #11.805574
vif_data = compute_vif(impt_features)

# remove X1 due to high multicolinearity w/ X1_X5
impt_features.remove('X1')#423.472807 
vif_data = compute_vif(impt_features)
print(vif_data)

X = trng_data[impt_features]

# fig = sns.pairplot(X)
# fig.savefig('finalPairPlot_x.png', dpi=300)

# fig = sns.pairplot(X) #BINGO!!
# fig.savefig('finalPairPlot_x_pt.png', dpi=300)

trng_data = X.assign(Y=y) 
# Transform our non-normal data
# skewed right Y so transform
trng_data['tY'], boxlambda = stats.boxcox(trng_data['Y'])
# print(trng_data['tY'])
# ty = pd.DataFrame(trng_data['tY'])

# X12', 'X1_X5', 'X4_X5', 'X5_X6' skewed
trng_data['log_Y'] = np.log(trng_data['Y'])
trng_data['inv_Y'] = 1/trng_data['Y']
trng_data['Y_sqrd'] = trng_data['Y']**2
Y_mean = np.mean(trng_data['Y'])
trng_data['center_Y'] = trng_data['Y'] - Y_mean

trng_data['log_X12'] = np.log(trng_data['X12'])
trng_data['inv_X12'] = 1/trng_data['X12']
X12_mean = np.mean(trng_data['X12'])
trng_data['center_X12'] = trng_data['X12'] - X12_mean

trng_data['log_X1_X5'] = np.log(trng_data['X1_X5'])
trng_data['inv_X1_X5'] = 1/trng_data['X1_X5']
X1_X5_mean = np.mean(trng_data['X1_X5'])
trng_data['center_X1_X5'] = trng_data['X1_X5'] - X1_X5_mean

trng_data['log_X4_X5'] = np.log(trng_data['X4_X5'])
trng_data['inv_X4_X5'] = 1/trng_data['X4_X5']
X4_X5_mean = np.mean(trng_data['X4_X5'])
trng_data['center_X4_X5'] = trng_data['X4_X5'] - X4_X5_mean

trng_data['log_X5_X6'] = np.log(trng_data['X5_X6'])
trng_data['inv_X5_X6'] = 1/trng_data['X5_X6']
X5_X6_mean = np.mean(trng_data['X5_X6'])
trng_data['center_X5_X6'] = trng_data['X5_X6'] - X5_X6_mean

y, X = dmatrices('tY ~ X6+X12+X1_X5+X4_X5+X5_X6+X5_X8', data = trng_data, return_type ='dataframe')
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 50, random_state = 42)
lin_reg = sm.OLS(y,X).fit()

linearity_test(lin_reg, y)
homoscedasticity_test(lin_reg)
normality_of_residuals_test(lin_reg)

acf = smt.graphics.plot_acf(lin_reg.resid, lags=25 , alpha=0.05)
acf.savefig('finalACF.png', dpi=300)

#Durbin Watson Test, the test statistic is between 0 and 4, <2 is positive correlation, >2 is negative correlation
# As a rule of thumb, anything between 1.5 and 2.5 is OK
DW = sms.durbin_watson(lin_reg.resid)
print(DW)
print(lin_reg.summary2())
calculate_rmse(lin_reg, test_x, test_y)

for column in X.columns:
    corr_test = stats.stats.pearsonr(X[column], lin_reg.resid)
    print(f'Variable: {column} --- correlation: {corr_test[0]:.4f}, p-value: {corr_test[1]:.4f}')

# print(lin_reg.params)
A = round(lin_reg.params['Intercept'], 4)
# print(A)
X6 = round(lin_reg.params['X6'], 4)
X12 = round(lin_reg.params['X12'], 4)
X1_X5 = round(lin_reg.params['X1_X5'], 4)
X4_X5 = round(lin_reg.params['X4_X5'], 4)
X5_X6 = round(lin_reg.params['X5_X6'], 4)
X5_X8 = round(lin_reg.params['X5_X8'], 4)

median_value = A + X6 + X12 + X1_X5 + X4_X5 + X5_X6 + X5_X8
print('median_value = ' + str(A) + ' + ' + str(X6) + ' + ' + str(X12) + 
    ' + ' + str(X1_X5) + ' + ' + str(X4_X5) + ' + ' + str(X5_X6) + ' + '+ str(X5_X8))

# Plot our model using Test Data to determine our y_pred
fig, ax = plt.subplots(figsize=(10,6))
fig.suptitle('Median Value Owner-Occupied Homes Test/Train Prediction Plot')
fig.tight_layout(pad=3)
plt.xlabel('Median Value (Test)', fontsize=10)
plt.ylabel('Median Value (Predict)', fontsize=10)
# subset our test data to align with our model
test_data = data.loc[456:, data.columns.isin(['Y', 'X6', 'X12', 'X1_X5', 'X4_X5', 'X5_X6', 'X5_X8'])]
# print(test_data)
X = test_data.loc[:, test_data.columns != 'Y']
y = test_data.loc[:, test_data.columns == 'Y']
X=sm.add_constant(X)

y_pred = lin_reg.predict(X)
# print(y_pred)
y_pred_df = pd.DataFrame(y_pred)
y_test_inv_boxcox = inv_boxcox(test_y, boxlambda)
y_pred_inv_boxcox = inv_boxcox(y_pred_df, boxlambda)
# print(y_pred_inv_boxcox)
sns.regplot(x=y_test_inv_boxcox,y=y_pred_inv_boxcox,ci=95,marker='o',color ='blue')
ax.grid()
fig.savefig('finalPredictionPlot.png', dpi=300)

#calculate prediction intervals
prediction=lin_reg.get_prediction(X)
predints=prediction.summary_frame(alpha=0.05)
obs_ci_lower = inv_boxcox(predints['obs_ci_lower'], boxlambda)
obs_ci_upper = inv_boxcox(predints['obs_ci_upper'], boxlambda)
# print(predints)

# Put the final predictions into a dataframe => 
# 'Census Tract’, ‘Prediction’, ‘Lower Prediction CI’, ‘Upper Prediction CI'
preds_df = pd.DataFrame(data.loc[456:, data.columns.isin(['Census Tract'])])
# adding lists as new column to dataframe df
preds_df['Prediction'] = y_pred_inv_boxcox
preds_df['Lower Prediction CI'] = obs_ci_lower
preds_df['Upper Prediction CI'] = obs_ci_upper
# converting to CSV file
preds_df.to_csv("Fawcett_Daniel.csv", encoding = 'utf-8', index = False)
