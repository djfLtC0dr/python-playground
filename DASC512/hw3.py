'''Dan Fawcett DASC 512 Homework 3'''
#Standard Import List
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat
from pathlib import Path
from statsmodels.stats import power
import statsmodels.api as sm
from statsmodels.stats import weightstats as stests

'''Problem 1a Right-Tail Test'''
placebo = (105, 119, 100, 97, 96, 101, 94, 95, 98)
mu = 100
alpha = 0.05
# usually dof = n - 1 for a single population sampling problem
dof = len(placebo) - 1
t_crit = stats.t.ppf(1 - alpha, dof)
t_stat, p_val = stats.ttest_1samp(placebo, mu)
# Since alternative hypothesis is one tailed, We need to divide the p value by 2.
p_val_upper_tail = float("{:.6f}".format(p_val/2)) 

print('T-Crit for this right-tailed placebo test is %f'%float("{:.6f}".format(t_crit)))
print('Test statistic is %f'%float("{:.6f}".format(t_stat)))
print('p-value for right-tailed test is %f'%p_val_upper_tail)

# Reject H0 only if the p_val < alpha (level of significance) or, 
# equivalently, t_stat were more extreme (further away from zero) t_crit.
if p_val_upper_tail <= alpha:
    print('Conclusion: Since p_val(=%f)'%p_val_upper_tail,'<=','alpha(=%.2f)'%alpha,
            'We reject the null hypothesis H0 (i.e. accept H_a)' + '\n' +
            'and conclude that placebo group RER population mean is greater than 100' + '\n'
            'i.e., placebo μ > 100 at %.2f level of significance'%alpha)
else:
    print('Conclusion: Since p_val(=%f)'%p_val_upper_tail,'>','alpha(=%.2f)'%alpha,
            'We fail to reject the null hypothesis H0 (i.e. accept H_0) '
            'and conclude the mean value placebo RER <= 100 at %.2f level of significance'%alpha)

'''Problem 1b Left-Tail Test'''
caffeine = (96, 99, 94, 89, 96, 93, 88, 105, 88)
mu = 92
# usually dof = n - 1 for a single population sampling problem
dof = len(caffeine) - 1
t_crit = -stats.t.ppf(alpha, dof)
t_stat, p_val = stats.ttest_1samp(caffeine, mu)
# p_value is wrong, it's for 2-sided, which is equivalent to Upper tail/2.  
# To convert subtract 1 and divide by 2
p_val_lower_tail= float("{:.6f}".format((1 - p_val) / 2))

print('T-Crit for this left-tailed caffeine test is %f'%float("{:.6f}".format(t_crit)))
print('T-stat statistic is %f'%float("{:.6f}".format(t_stat)))
print('p-value for left-tailed test is %f'%p_val_lower_tail)

if p_val_lower_tail <= alpha:
    print('Conclusion: Since p_val(=%f)'%p_val_lower_tail,'<=','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0. 
            (i.e. accept H_a) and conclude that Caffeine shows a negative effect on RER 
            i.e., μ < 92 at %.2f level of significance'''%alpha)
else:
    print('Conclusion: Since p_val(=%f)'%p_val_lower_tail,'>','alpha(=%.2f)'%alpha,'''We fail to reject the null hypothesis H0 
            (i.e. accept H_0) and conclude there is no significant mean difference in RER
            i.e., caffeine μ = 92 at %.2f level of significance'''%alpha)            

'''Problem 1c 2-sided test'''
dof = (len(placebo) - 1) + (len(caffeine) - 1)
t_crit = stats.t.ppf((1-alpha)/2, dof)
print(np.std(placebo, ddof=1), np.std(caffeine, ddof=1))
 # out => 7.699206308300732 5.6075346137535735
 # Different stdev so need to use equal_var = False testing for zero order doesn't matter
t_stat, p_val = stats.ttest_ind(placebo, caffeine, equal_var = False)

print('T-Crit for this two-tailed test is %f'%float("{:.6f}".format(t_crit)))
print('Test statistic is %f'%float("{:.6f}".format(abs(t_stat))))
print('p-value for two-tailed test is %f'%p_val)

if p_val <= alpha:
    print('Conclusion: Since p_val(=%f)'%p_val,'<=','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0. 
            (i.e. accept H_a) and conclude that there is a significant difference in RER 
            between the mean value RER for the Placebo and Caffeine groups.
            i.e. μ > 0 at %.2f level of significance'''%alpha)
else:
    print('Conclusion: Since p_val(=%f)'%p_val,'>','alpha(=%.2f)'%alpha,'''We fail to reject the null hypothesis H0 
            (i.e. accept H_0) and conclude these means are statistically equal 
            at %.2f level of significance'''%alpha) 

'''Problem 2 Right-tale test'''
# Estimations of plasma calcium concentration in 18 patients with Everley’s syndrome gave a mean of
# 3.2 mmol/l, with standard deviation 1.1. Previous experience from a number of investigations and 
# published reports had shown that the mean was commonly close to 2.5 mmol/l in healthy people aged 
# 20-44, the age range of the patients, and is normally distributed. 
# Is the mean in these patients abnormally high at a significance level of 0.05?
n = 18
x1 = 3.2
mu = 2.5
x1_stdev = 1.1
std_error_x1 = x1_stdev / sqrt(n)
mean_diff = mu - x1
alpha = 0.05
# usually dof = n - 1 for a single population sampling problem
dof = n - 1
t_stat = (mean_diff / std_error_x1)
# p_val = stats.t.cdf(t_stat, dof)
t_crit = stats.t.ppf(1 - alpha, dof)
p_val = 1 - stats.norm.cdf(t_crit)

print('T-Crit for this right-tailed plasma test is %f'%float("{:.6f}".format(t_crit)))
print('Test statistic is %f'%float("{:.6f}".format((t_stat))))
print('p-value for right-tailed plasma test is %f'%p_val)

'''Problem 3'''
co_1 = (103, 94, 110, 87, 98)
co_2 = (97, 82, 123, 92, 175, 88, 118)
alpha = 0.10

dof = (len(co_1) - 1) + (len(co_2) - 1)
t_crit = stats.t.ppf(alpha/2, dof)
print(np.std(co_1, ddof=1), np.std(co_2, ddof=1))
 # out => 8.734987120768983 32.185474393035776
 # Different stdev so need to use equal_var = False testing for zero order doesn't matter
t_stat, p_val = stats.ttest_ind(co_1, co_2, equal_var = False)

print('T-Crit for this two-tailed run-time test is %f'%float("{:.6f}".format(t_crit)))
print('Test statistic is %f'%float("{:.6f}".format(abs(t_stat))))
print('p-value for two-tailed run-time test is %f'%p_val)

'''Problem 5'''
# mean breaking strength of 15 kilograms with a standard deviation of 0.5 kilograms. 
# Assume that his estimate of standard deviation is correct (sd known) and that with α = 0.05 
# and π = 0.90 determine the sample size required to test the hypothesis that 
# µ ≥ 15 if the true value is µ = 14.8 or µ = 14.9.
x1 = 15
sigma = 0.5
alpha = 0.05
# Margin or Error
E = 14.9 - 14.8
# Determine the z-critical value
z_crit = stats.norm.ppf(1 - alpha)
n = ((z_crit * sigma) / E)**2
print(n)

'''Problem 6'''
df_bavg = pd.read_csv("BattingAverages.csv", sep = ',')
# Clean-up
df_bavg.drop('Unnamed: 6', axis=1, inplace=True)
# print(df_bavg.head())
size = len(df_bavg.index)
# print('bavg length: ' + str(size))
mu, sigma = np.mean(df_bavg['BattingAvg']), np.std(df_bavg['BattingAvg'])
# print('mu bat avg = ', mu)
means = np.random.normal(mu, sigma, size)

# Histplot
fig, ax = plt.subplots(figsize=(6,4))
fig.tight_layout(pad = 3)
fig.suptitle('Batting Avg Histplot') 
#create normal distribution curve
sns.histplot(means, kde=True)
# sns.histplot(data=means, bins=bins, ax=ax_hist, color='green', kde=True, stat="count", linewidth=0)
# Use scipy.stats implementation of the normal pdf
# Plot the distribution curve
x = np.linspace(0, 0.5, num=size)
plt.plot(x, stats.norm.pdf(x, mu, sigma), color='green')

# Boxplot
fig, ax = plt.subplots(figsize=(6,4))
fig.tight_layout(pad = 3)
fig.suptitle('Batting Avg Boxplot') 
sns.boxplot(data=means, color = 'darkgreen')

# QQ Plot
fig, ax = plt.subplots(figsize=(6,4))
fig.suptitle('Batting Avg QQ-Plot')
fig.tight_layout(pad=3)
pp = sm.ProbPlot(np.array(means), stats.norm, fit=True)
qq = pp.qqplot(marker='.', ax=ax, markerfacecolor='darkorange', markeredgecolor='darkorange', alpha=0.8)
sm.qqline(qq.axes[0], line='45', fmt='k--')
# plt.show()
stat, pval = stats.normaltest(means)
print('Statistics=%.3f, p=%.3f' % (stat, pval))
# interpret
alpha = 0.05
if pval > alpha:
	print('Sample looks Normal (fail to reject H0)')
else:
	print('Sample does not look Normal (reject H0)')

df_bavg['BattingAvg'].describe()
zstat, pval = stests.ztest(df_bavg['BattingAvg'], x2=None, value=0.265)
print('zstat=%.3f, p=%.3f' % (zstat, pval))
if pval < alpha:
  print("Null hyphothesis rejected , Alternative hyphothesis accepted conclude mean batting avg < 0.265")
else:
  print("Null hyphothesis accepted , Alternative hyphothesis rejected")

# perform power analysis
analysis = power.TTestIndPower()
power = analysis.solve_power(power=None, effect_size=1, alpha=0.05, nobs1=10, ratio=1.0)
print('Power: %.3f' % power)

df_ba_nl = df_bavg.loc[df_bavg['League'] == 'National League']['BattingAvg']
df_ba_al = df_bavg.loc[df_bavg['League'] == 'American League']['BattingAvg']
print(np.std(df_ba_nl, ddof=1), np.std(df_ba_al, ddof=1))
# out => 0.033159408525780996 0.03484792169158821
# = stdev so need to use equal_var = True testing for zero order doesn't matter
t_stat, p_val = stats.ttest_ind(df_ba_nl, df_ba_al, equal_var = True)
print('t_stat=%.3f, p=%.3f' % (stat, p_val))
# interpret
alpha = 0.05
if p_val > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')

'''Problem 7'''
df_bt = pd.read_csv("BodyTemp.csv", sep = ',')
# Clean-up
df_bt.drop('Unnamed: 2', axis=1, inplace=True)
# print(df_bt.head())
size = len(df_bt.index)
# print('bavg length: ' + str(size))
mu, sigma = np.mean(df_bt['BodyTemp']), np.std(df_bt['BodyTemp'])
# print('mu body temp = ', mu)
means = np.random.normal(mu, sigma, size)

# Histplot
fig, ax = plt.subplots(figsize=(6,4))
fig.tight_layout(pad = 3)
fig.suptitle('Body Temp Histplot') 
#create normal distribution curve
sns.histplot(means, kde=True)
# sns.histplot(data=means, bins=bins, ax=ax_hist, color='green', kde=True, stat="count", linewidth=0)
# Use scipy.stats implementation of the normal pdf
# Plot the distribution curve
x = np.linspace(96, 102, num=size)
plt.plot(x, stats.norm.pdf(x, mu, sigma), color='green')

# Boxplot
fig, ax = plt.subplots(figsize=(6,4))
fig.tight_layout(pad = 3)
fig.suptitle('Body Temp Boxplot') 
sns.boxplot(data=means, color = 'darkgreen')

# QQ Plot
fig, ax = plt.subplots(figsize=(6,4))
fig.suptitle('Body Temp QQ-Plot')
fig.tight_layout(pad=3)
pp = sm.ProbPlot(np.array(means), stats.norm, fit=True)
qq = pp.qqplot(marker='.', ax=ax, markerfacecolor='darkorange', markeredgecolor='darkorange', alpha=0.8)
sm.qqline(qq.axes[0], line='45', fmt='k--')
# plt.show()

stat, pval = stats.normaltest(means)
print('Statistics=%.3f, p=%.3f' % (stat, pval))
# interpret
alpha = 0.05
if pval > alpha:
	print('Sample looks Normal (fail to reject H0)')
else:
	print('Sample does not look Normal (reject H0)')

df_bt['BodyTemp'].describe()
alpha = 0.20
zstat, pval = stests.ztest(df_bt['BodyTemp'], value=98.6)
print('zstat=%.3f, p=%.3f' % (zstat, pval))
if pval < alpha:
  print("Null hyphothesis rejected , Alternative hyphothesis accepted")
else:
  print("Null hyphothesis accepted , Alternative hyphothesis rejected")

# perform power analysis
power = analysis.solve_power(power=None, effect_size=1, alpha=alpha, nobs1=10, ratio=1.0)
print('Power: %.3f' % power)

df_bt_male = df_bt.loc[df_bt['Gender'] == 'Male']['BodyTemp']
df_bt_female = df_bt.loc[df_bt['Gender'] == 'Female']['BodyTemp']
print(np.std(df_bt_male, ddof=1), np.std(df_bt_female, ddof=1))
# out => 0.7008272469467351 0.7414338650419362
# = stdev equal, checking for mean diff
t_stat, p_val = stests.ztest(df_bt_male, df_bt_female, value = 0)
print('t_stat=%.3f, p=%.3f' % (stat, p_val))
# interpret
alpha = 0.05
if p_val > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0), conclude the mean Body Temps are  significantly different between males & females')

'''Problem 8'''
# perform power analysis
# If I want to detect an effect size of 1 (number of SD from the mean) 
# and want α = 0.1 and π = 0.8, how large of a sample should I collect?
sample_size = analysis.solve_power(power=0.8, effect_size=1, alpha=0.1, nobs1=None, ratio=1.0)
print('Sample Size: %.3f' % sample_size)
