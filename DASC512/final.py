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
# print(df_student.head())

# explore data.
# df_student.info()

# create the training data
y_column = ['Y']
y = data[y_column]
x_columns = ['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X1*X5','X4*X5','X5*X6','X5*X8']
x = data[x_columns]

# remove features w/ variance < 30% => features which mostly remain at the same level 
# across different observations, should not ideally be responsible for differing responses in the observations.   
var = VarianceThreshold(threshold=0.3)
var = var.fit(x,y)
cols = var.get_support(indices=True)
features = x.columns[cols]
# print(features)
x = data[features]
# print(x)

# re-assign our df w/ only the features w/ variance > 30% + our target variable
data = x.assign(median_value=y['Y']) 
# print(data)

# Remove features which are not correlated with the response variable 
plt.figure(figsize=(12,12))
cor = data.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()

# Consider correlations only with the target variable
cor_target = abs(cor['median_value'])

#Select correlations with a correlation above a threshold 10%.
features = cor_target[cor_target>0.1]
print(features.index)
x_columns = ['Ed', 'Po1', 'Po2', 'M.F', 'Pop', 'U2', 'Wealth', 'Ineq', 'Time']

# Figure out the multicollinearity features to remove via VIF
def compute_vif(considered_features):
    X = data[considered_features]
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
    x = data[x_columns]
    results = sm.OLS(y, x).fit()
    print(results.summary2())
    return results
get_stats()

model_median_value = get_stats()

# TODO remove the least statistically significant variable(s) i.e. pval > 0.05
x_columns.remove('') # pval 0.5860
model_crime=get_stats()

# TODO # Run residual analysis (graphically) to determine if model is accurate.
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
# fig.savefig('crime_residuals_fitted.png', dpi=300)

#Normality Plots
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},figsize=(12,8))
fig.suptitle('Normality Plots')
sns.boxplot(data=residuals, ax=ax_box, color='darkorchid')
sns.histplot(data=residuals, ax=ax_hist, color='orchid')
ax_box.set(xlabel='')
fig.savefig('crime_normality_plot.png', dpi=300)

# QQ Plot
fig, ax = plt.subplots(figsize=(6,4))
fig.suptitle('Crime QQ-Plot')
fig.tight_layout(pad=3)
pp = sm.ProbPlot(residuals, stats.norm, fit=True)
qq = pp.qqplot(marker='.', ax=ax, markerfacecolor='darkorange', markeredgecolor='darkorange', alpha=0.8)
sm.qqline(qq.axes[0], line='45', fmt='k--')
# fig.savefig('crime_qq_plot.png', dpi=300)

# Determine Regression 
ols = LinearRegression()
X = data[x_columns]
model = ols.fit(X, y)
# print(model.coef_)
# print(model.intercept_)
# print(model.score(X, y))
A = model.intercept_[0]
# print(A)
X1 = model.coef_[0][0]
# print(X1)
X2 = model.coef_[0][1]
# print(X2)
crime = A + X1 + X2
print('crime = ' + str(A) + ' + ' + str(X1) + ' + ' + str(X2) + ' => ' , crime)

# Plot our model using Test/Train Data
fig, ax = plt.subplots(figsize=(6,4))
fig.suptitle('Median Value Owner-Occupied Homes Test/Train Prediction Plot')
fig.tight_layout(pad=3)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
linreg=LinearRegression()
linreg.fit(x_train,y_train)
y_pred=linreg.predict(x_test)
sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');
# fig.savefig('crime_test_train_predict_plot.png', dpi=300)