import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from statistics import NormalDist
import seaborn as sns
from pathlib import Path
from statsmodels.stats import power
import statsmodels.api as sm
from statsmodels.stats import weightstats as stests
from math import isqrt

'''Problem 1'''
# Suppose that the four inspectors @film factory are supposed 
# to stamp the expiration date on each package of film @EO the assembly line. 
# John, who stamps 20% of the packages, fails to stamp the expiration date once in every 200 packages; 
# Tom, who stamps 60% of the packages, fails to stamp the expiration date once in every 100 packages; 
# Jeff, who stamps 15% of the packages, fails to stamp the expiration date once in every 90 packages; 
# Pat, who stamps 5% of the packages fails to stamp the expiration date once in every 200 packages. 
# If a customer complains that her package of film does not show the expiration date, 
# what is the probability that it was inspected by John?

# the pkg is marked by John
prob_john = (.20)*(1/200)
# print(prob_John)
# the pkg is marked by Tom
prob_tom = (.60)*(1/100)
# the pkg is marked by Jeff
prob_jeff = (.15)*(1/90)
# the pkg is marked by Pat
prob_pat =  (.05)*(1/200)
# the pkg is not marked
prob_no_exp_dt = prob_john + prob_tom + prob_jeff + prob_pat
# print(prob_no_exp_dt)
# No printed expiration date inspected by John
prob_john_no_exp_dt = (prob_john/prob_no_exp_dt)
# print('prob_no_exp_dt_John = %.4f' % prob_no_exp_dt_John)

'''Problem 2'''
A = (2, 4, 5, 7)
B = (1, 3, 4, 7)
C = (2, 2, 2, 5)
sum_C = sum(C)
# print(sum_C)
human_error = C[3]
# print(human_error)
prob_he_int_c = human_error/sum_C
# print('prob_he_int_c = %.4f' % prob_he_int_c)

'''Problem 3'''
df_of = pd.read_csv("faithful.csv", sep = ',')
# print(df_of.head())
eruptions = df_of['eruptions']
waiting = df_of['waiting']
fig, ax = plt.subplots(figsize=(6,4))
fig.tight_layout(pad = 3)
plt.scatter(x=eruptions, y=waiting)
plt.title("Eruptions vs Wait Times for Ol' Faithful")
plt.xlabel("Eruptions)")
plt.ylabel("Wait Times")

'''Problem 3b Left-Tail Test'''
# only want data eruptions < 3 mins
df_of_lt3 = df_of.loc[df_of['eruptions'] < 3]
# print(df_of_lt3)
alpha = 0.05
# usually dof = n - 1 for a single population sampling problem
dof = len(df_of_lt3['eruptions']) - 1
# print(dof)
t_crit = -stats.t.ppf(alpha, dof)
tstat, pval = stests.ztest(df_of_lt3['waiting'], x2=None, value=60, ddof=dof)
print('tstat=%.4f, pval=%.4f' % (tstat, pval))
# p_value is wrong, it's for 2-sided, which is equivalent to Upper tail/2.  
# To convert subtract 1 and divide by 2
p_val_lower_tail= float("{:.6f}".format((1 - pval) / 2))

print('T-Crit for this left-tailed test is %f'%float("{:.6f}".format(t_crit)))
print('T-stat statistic is %f'%float("{:.6f}".format(tstat)))
print('p-value for left-tailed test is %f'%p_val_lower_tail)

if p_val_lower_tail < alpha:
    print('Conclusion: Since pval(=%f)'%p_val_lower_tail,'<','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0. 
            (i.e. accept H_a) and conclude mean wait time avg < 60
            i.e., μ < 60 at %.2f level of significance'''%alpha)
else:
    print('Conclusion: Since pval(=%f)'%p_val_lower_tail,'>','alpha(=%.2f)'%alpha,'''We fail to reject the null hypothesis H0 
            (i.e. accept H_0) and conclude mean wait time avg >= 60
            i.e., μ >= 60 at %.2f level of significance'''%alpha) 

'''Problem 3c Power'''
analysis = power.TTestIndPower()
sample_size = len(df_of_lt3['eruptions'])
# print('sample_size: ', sample_size)
pwr = analysis.solve_power(effect_size=0.5, alpha=alpha, nobs1=sample_size)
print('Power: %.3f' % pwr)

'''Problem 3d Confidence Interval for df_of_lt3'''
def confidence_interval(data, confidence=0.95):
  dist = NormalDist.from_samples(data)
  z = NormalDist().inv_cdf((1 + confidence) / 2.)
  h = dist.stdev * z / ((len(data) - 1) ** .5)
  return dist.mean - h, dist.mean + h

print(confidence_interval(df_of_lt3['waiting']))

'''Problem 3e another left tail test this time for df_of_gte3'''
# only want data eruptions >= 3 mins
df_of_gte3 = df_of.loc[df_of['eruptions'] >= 3]
# print(df_of_gte3)
alpha = 0.05
# usually dof = n - 1 for a single population sampling problem
dof = len(df_of_gte3['eruptions']) - 1
t_crit = -stats.t.ppf(alpha, dof)
tstat, pval = stests.ztest(df_of_gte3['waiting'], x2=None, value=80, ddof=dof)
print('tstat=%.4f, pval=%.4f' % (tstat, pval))
# p_value is wrong, it's for 2-sided, which is equivalent to Upper tail/2.  
# To convert subtract 1 and divide by 2
p_val_lower_tail= float("{:.6f}".format((1 - pval) / 2))

print('T-Crit for this left-tailed test is %f'%float("{:.6f}".format(t_crit)))
print('T-stat statistic is %f'%float("{:.6f}".format(tstat)))
print('p-value for left-tailed test is %f'%p_val_lower_tail)

if p_val_lower_tail < alpha:
    print('Conclusion: Since pval(=%f)'%p_val_lower_tail,'<','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0. 
            (i.e. accept H_a) and conclude mean wait time avg > 80 mins
            i.e., μ < 80 at %.2f level of significance'''%alpha)
else:
    print('Conclusion: Since pval(=%f)'%p_val_lower_tail,'>','alpha(=%.2f)'%alpha,'''We fail to reject the null hypothesis H0 
            (i.e. accept H_0) and conclude mean wait time avg < 80 mins
            i.e., μ >= 80 at %.2f level of significance'''%alpha) 

'''Problem 3f Power'''
sample_size = len(df_of_gte3['eruptions'])
# print('sample_size: ', sample_size)
pwr = analysis.solve_power(effect_size=0.5, alpha=alpha, nobs1=sample_size)
print('Power: %.3f' % pwr)

'''Problem 3g Conf Interval for df_of_gte3'''
print(confidence_interval(df_of_gte3['waiting']))

'''Problem 3h 2x boxplots same axis for means <3 & >=3 '''
mu_lt3, sigma_lt3 = np.mean(df_of_lt3['waiting']), np.std(df_of_lt3['waiting'])
sample_size_lt3 = len(df_of_lt3['waiting'])
means_lt3 = np.random.normal(mu_lt3, sigma_lt3, sample_size_lt3)
mu_gte3, sigma_gte3 = np.mean(df_of_gte3['waiting']), np.std(df_of_gte3['waiting'])
sample_size_gte3 = len(df_of_gte3['waiting'])
means_gte3 = np.random.normal(mu_gte3, sigma_gte3, sample_size_gte3)
fig, axs = plt.subplots(1, 2, figsize=(12,8), tight_layout = True)
sns.boxplot(data=means_lt3, color = 'darkgreen', ax=axs[0])
sns.boxplot(data=means_gte3, color = 'lightgreen', ax=axs[1])
# sns.boxplot(x='eruptions', y='waiting', data=df_of_lt3, )
# sns.boxplot(x='eruptions',y='waiting', data=df_of_gte3, ax=axs[1])
fig.suptitle('Boxplots Eruptions < 3-mins & >= 3-mins')

'''Problem 3i ztest b/t means 20 mins'''
# checking for mean diff = 20
tstat, pval = stests.ztest(df_of_lt3['waiting'], df_of_gte3['waiting'], value = 20)
print('tstat=%.4f, pval=%.4f' % (tstat, pval))
# interpret
alpha = 0.05
if pval < alpha:
    print('Conclusion: Since pval(=%f)'%pval,'<','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0. 
            (i.e. accept H_a) and conclude the mean Wait Times are significantly different than 20 mins for eruptions <3-mins & >= 3-mins
            i.e., μ != 20 at %.2f level of significance'''%alpha)
else:
    print('Conclusion: Since pval(=%f)'%p_val_lower_tail,'>','alpha(=%.2f)'%alpha,'''We fail to reject the null hypothesis H0 
            (i.e. accept H_0) and conclude mean Wait Times 
            = 20 mins for eruptions <3-mins & >= 3-mins
            i.e., μ >= 80 at %.2f level of significance'''%alpha) 

'''Problem 3j Power'''
sample_size = len(df_of_lt3['waiting']) + len(df_of_gte3['waiting'])
# print('sample_size: ', sample_size)
pwr = analysis.solve_power(effect_size=0.5, alpha=0.05, nobs1=sample_size)
print('Power: %.3f' % pwr)  

'''Problem 3k Confidence Interval for 2x df mean wait times'''
# concat the two DataFrames on top of each other
df_stacked_waits = pd.concat([df_of_lt3['waiting'], df_of_gte3['waiting']], axis=0)
print(confidence_interval(df_stacked_waits))

'''Problem 3l histogram for the wait times'''
# Creating histogram
fig, ax = plt.subplots(1, 1, figsize =(12, 8), tight_layout = True)
plt.xlabel("Wait Times")
plt.ylabel("Count")    
plt.title("Histogram Eruptions < 3-mins & >= 3-mins'")
hist_wt_bins = isqrt(len(df_stacked_waits))             
hist_wait_times = ax.hist(df_stacked_waits, bins = hist_wt_bins)

'''Problem 3m QQ-Plot to for wait times to assess normality.'''
mu_combo_wt, sigma_combo_wt = np.mean(df_stacked_waits), np.std(df_stacked_waits)
sample_size_combo_wt = len(df_stacked_waits)
means_combo_wt = np.random.normal(mu_combo_wt, sigma_combo_wt, sample_size_combo_wt)
fig, ax = plt.subplots(figsize=(6,4))
fig.suptitle('Wait Times QQ-Plot')
fig.tight_layout(pad=3)
pp = sm.ProbPlot(np.array(means_combo_wt), stats.norm, fit=True)
qq = pp.qqplot(marker='.', ax=ax, markerfacecolor='darkorange', markeredgecolor='darkorange', alpha=0.8)
sm.qqline(qq.axes[0], line='45', fmt='k--')

'''Problem 3n Perform a test for normality on the wait times.'''
tstat, pval = stats.normaltest(means_combo_wt)
print('tstat=%.4f, p=%.4f' % (tstat, pval))
# interpret
alpha = 0.05
if pval < alpha:
    print('Since pval(=%.4f)'%pval,'<','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0 and conclude the Sample does not look Normal''')
else:
	print('Since pval(=%.4f)'%pval,'>','alpha(=%.2f)'%alpha,'''We fail to reject the null hypothesis H0 and conclude the Sample looks Normal''')
