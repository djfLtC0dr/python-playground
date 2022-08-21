import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
# from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

# Read the data 
data = pd.read_csv("student_data.csv", sep = ',')
# print(data)

# explore data.
# data.info()

# subset trng data to only data with Y values => first 456 records
subset_data = data[:456]
# subset_data.info()

# subset our test data
test_data = data.tail(50)
# drop unnecessary columns from our test dataset
test_data.drop(['Census Tract', 'Y'], axis=1, inplace = True)
# print(test_data)

# create the training data
y_column = ['Y']
y = subset_data[y_column]
x_columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X1*X5','X4*X5','X5*X6','X5*X8']
x = subset_data[x_columns]

# remove features w/ variance < 30% => features which mostly remain at the same level 
# across different observations, should not ideally be responsible for differing responses in the observations.   
var = VarianceThreshold(threshold=0.3)
var = var.fit(x,y)
cols = var.get_support(indices=True)
features = x.columns[cols]
# print(features)
x = subset_data[features]
# print(x)

# re-assign our df w/ only the features w/ variance > 30% + our target variable
subset_data = x.assign(median_value=y['Y']) 
# print(subset_data)

# Remove features which are not correlated with the response variable 
# plt.figure(figsize=(12,12))
cor = subset_data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

# Consider correlations only with the target variable
cor_target = abs(cor['median_value'])

#Select correlations with a correlation above a threshold 10%.
features = cor_target[cor_target>0.1]
# print(features.index)
x_columns = ['X1', 'X2', 'X3', 'X5', 'X6', 'X7', 'X9', 'X10', 'X11', 'X12', 'X1*X5',
            'X4*X5', 'X5*X8']

# Figure out the multicollinearity features to remove via VIF
def compute_vif(considered_features):
    X = subset_data[considered_features]
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

## creating function to get model statistics
def get_stats():
    x = subset_data[x_columns]
    results = sm.OLS(y, x).fit()
    print(results.summary2())
    return results

# model_median_value = get_stats()

# remove the least statistically significant features(s) i.e. pval > 0.05
x_columns.remove('X5') # pval 0.8135
x_columns.remove('X2') # pval 0.4938
x_columns.remove('X9') # pval 0.4491
x_columns.remove('X7') # pval 0.1492
x_columns.remove('X3') # pval 0.0570
model_median_value=get_stats()

# drop unnecessary columns from our test dataset
# all except x_columns => ['X6', 'X11', 'X12', 'X1*X5', 'X4*X5', 'X5*X8']
# x_columns = ['X1', 'X2', 'X3', 'X5', 'X6', 'X7', 'X9', 'X10', 'X11', 'X12', 'X1*X5',
#             'X4*X5', 'X5*X8']
test_data.drop(['X1', 'X2', 'X3', 'X4', 'X5', 'X7', 'X8', 'X9','X10', 'X5*X6'], axis=1, inplace = True)
# print(test_data)

# Run residual analysis (graphically) to determine if model is accurate.
#Pull residuals
residuals = model_median_value.resid
fitted_values = model_median_value.fittedvalues

#plot Residuals vs Fitted Values
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(fitted_values, residuals, alpha=1.0, color='red')
fig.suptitle('Residuals versus Fitted Values - Median Value Owner-Occupied Homes')
plt.ylabel("Residual")
plt.xlabel("Fitted Values")
fig.tight_layout(pad=3)
ax.grid(True)
plt.axvline(x=min(fitted_values)+(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axvline(x=min(fitted_values)+2*(max(fitted_values)-min(fitted_values))/3, color='darkblue')
plt.axhline(y=0,color='black')
# fig.savefig('median_value_residuals_fitted.png', dpi=300)

#Normality Plots
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(12,8))
fig.suptitle('Normality Plots')
sns.boxplot(data=residuals, ax=ax_box, color='darkorchid')
sns.histplot(data=residuals, ax=ax_hist, color='orchid')
ax_box.set(xlabel='')
# fig.savefig('median_value_normality_plot.png', dpi=300)

# QQ Plot
fig, ax = plt.subplots(figsize=(6,4))
fig.suptitle('Median_Value QQ-Plot')
fig.tight_layout(pad=3)
pp = sm.ProbPlot(residuals, stats.norm, fit=True)
qq = pp.qqplot(marker='.', ax=ax, markerfacecolor='darkorange', markeredgecolor='darkorange', alpha=0.8)
sm.qqline(qq.axes[0], line='45', fmt='k--')
# fig.savefig('crime_qq_plot.png', dpi=300)

# Determine Regression 
ols = LinearRegression()
X = subset_data[x_columns]
model = ols.fit(X, y)
# print(model.coef_)
# print(model.intercept_)
# print(model.score(X, y))
A = model.intercept_[0]
# print(A)
X6 = model.coef_[0][0]
X11 = model.coef_[0][1]
X12 = model.coef_[0][2]
X1_X5 = model.coef_[0][3]
X4_X5 = model.coef_[0][4]
X5_X8 = model.coef_[0][5]

median_value = A + X11 + X12 + X1_X5 + X4_X5 + X5_X8
print('median_value = ' + str(A) + ' + ' + str(X11) + ' + ' + str(X12) + 
    ' + ' + str(X1_X5) + ' + ' + str(X4_X5) + ' + ' + str(X5_X8) +' => ' , median_value)

# Plot our model using Test/Train Data
fig, ax = plt.subplots(figsize=(6,4))
fig.suptitle('Median Value Owner-Occupied Homes Test/Train Prediction Plot')
fig.tight_layout(pad=3)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=50,random_state=42)
linreg=LinearRegression()
linreg.fit(x_train,y_train)
y_pred=linreg.predict(test_data)
sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');
# fig.savefig('crime_test_train_predict_plot.png', dpi=300)

#TODO: Confidence intervals
