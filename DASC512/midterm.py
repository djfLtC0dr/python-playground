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
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import(pairwise_tukeyhsd, MultiComparison)

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

'''Problem 5 IQ Power @AFIT'''
# IQ of incoming students at AFIT. We desire to run a test that will detect a 5 IQ point difference 
# between the true mean and our new class. From previous research, we assume a standard deviation of 15 (sd is known) 
# and leadership wants to have options for α = 0.05 and α = 0.1. Create a plot for a power analysis 
# where the y axis is the power of our test and the x axis is the sample size. 
# It should have a line for α = 0.05 and a line for α = 0.1
x1 = 108
x2 = 112
sigma = 15
mu = 100
effect_size1 = (x1 - mu)/sigma
effect_size2 = (x2 - mu)/sigma
sample_sizes = np.array([25, 50])
alphas = np.array([0.05, 0.10])
effect_sizes = np.array([effect_size1, effect_size2])
plt.style.use('seaborn')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig = analysis.plot_power(
    dep_var='nobs', nobs=sample_sizes,  
    effect_size=effect_sizes, alpha=alphas, ax=ax, 
    title='Power Analysis\n$\\alpha$ = [0.05, 0.10]')
fig.tight_layout(pad=3)
plt.xlabel("Number of Observations")
plt.ylabel("Power")
ax.grid(True)

'''Problem 6 Precision Air Drop Probability'''
# The accuracy of a new precision air drop system being tested by the US Air Force 
# follows a normal distribution with a mean of 50 ft and a standard deviation of 10 feet. 
# A particular resupply mission is considered successful if at least 6 of the 9 payloads are 
# delivered with an accuracy of between 40 and 55 feet. 
# What is the probability that the resupply mission will be successful?
mu_pads, sigma_pads = 50, 10
upper_prob = stats.norm(mu_pads, sigma_pads).cdf(55)
lower_prob = stats.norm(mu_pads, sigma_pads).cdf(40)
# SUBTRACT the probabilities (upper - lower) we're looking for the probability (area under the curve) 
prob_success = upper_prob - lower_prob
# now we need to get the set of probabilities of discrete outcomes
x, n = 6, 9
msn_success_prob = stats.binom.pmf(x, n, prob_success)
print(msn_success_prob) #=> 0.19596938728722993

'''Problem 7 Does sigma of rods exceeds 3.3'''
# Import data file into a dataframe
txt_file = "steel.txt"
COLUMN_NAMES=['steel']

steel_data = pd.read_table(txt_file, delim_whitespace=True, header=None, names=COLUMN_NAMES,
                          lineterminator='\n')
# print(steel_data.head())
steel_std = np.std(steel_data['steel'])
print('Steel stdev: %.4f'%steel_std)

'''Problem 8 Left Tail T-Test the mean chlorine content is =71ppm vs <71ppm '''
# Import data file into a dataframe
txt_file = "chlorine.txt"
COLUMN_NAMES=['cl_sample']

cl_data = pd.read_table(txt_file, delim_whitespace=True, header=None, names=COLUMN_NAMES,
                          lineterminator='\n')
# print(cl_data.head())
mu = 71
alpha = 0.05
# usually dof = n - 1 for a single population sampling problem
dof = len(cl_data) - 1
t_crit = -stats.t.ppf(alpha, dof)
res = stats.ttest_1samp(cl_data, mu)
tstat, pval = res.statistic, res.pvalue
# p_value is wrong, it's for 2-sided, which is equivalent to Upper tail/2.  
# To convert subtract 1 and divide by 2
p_val_lower_tail= float("{:.6f}".format((1 - pval[0]) / 2))

print('T-Crit for this left-tailed caffeine test is %f'%float("{:.4f}".format(t_crit)))
print('T-stat statistic is %f'%float("{:.4f}".format(tstat[0])))
print('p-value for left-tailed test is %f'%p_val_lower_tail)

if p_val_lower_tail <= alpha:
    print('Conclusion: Since p_val(=%f)'%p_val_lower_tail,'<=','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0. 
            (i.e. accept H_a) and conclude that mean chlorine content is less than 71 ppm 
            i.e., μ < 71ppm at %.2f level of significance'''%alpha)
else:
    print('Conclusion: Since p_val(=%f)'%p_val_lower_tail,'>','alpha(=%.2f)'%alpha,'''We fail to reject the null hypothesis H0 
            (i.e. accept H_0) and conclude mean chlorine content is equal to 71 ppm
            i.e., μ = 71ppm at %.2f level of significance'''%alpha)    

'''Problem 9 ANOVA + Hyp Test dental_crown.txt for significance'''
# Column 1 is the dentist who applied the crown, 
# column 2 is the method used, 
# column 3 is the alloy used, 
# column 4 is the temperature of the application instrument and 
# column 5 is the response (diamond pyramid hardness). 
# Construct an ANOVA table and perform hypothesis tests to determine which (if any) factors 
# and interactions are significant.
# Import data file into a dataframe
txt_file = "dental_crown.txt"
COLUMN_NAMES=['dentist','method','alloy','temp','hardness']

dc_data = pd.read_table(txt_file, delim_whitespace=True, header=None, names=COLUMN_NAMES,
                          lineterminator='\n')
# print(dc_data.head())
# # generate a boxplot to see the data distribution by effect. 
fig, ax = plt.subplots(figsize=(6,4))
fig.tight_layout(pad = 3)
fig.suptitle('Factors vs. Hardness Box Plot') 
ax = sns.boxplot(x='method', y='hardness', data=dc_data, color='#99c2a2')
ax = sns.boxplot(x='alloy', y='hardness', data=dc_data, color='#99c2a2')
ax = sns.boxplot(x='temp', y='hardness', data=dc_data, color='#99c2a2')

# Ordinary Least Squares (OLS) model
formula = 'hardness ~ method + alloy + temp + method:alloy:temp'
model = ols(formula, data=dc_data).fit()
anova_table = anova_lm(model, typ=2)
print(anova_table)

# Multicomp Tukey
interaction_groups = "Method_" + dc_data['method'].astype(str) + " & " + "Alloy_" + dc_data['alloy'].astype(str) + " & " + "Temp_" + dc_data['temp'].astype(str)
multi_comp = MultiComparison(dc_data['hardness'], interaction_groups)
print(multi_comp.tukeyhsd().summary())

from matplotlib.backends.backend_pdf import PdfPages
# Save all figures to PDF
def save_figs_pdf(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

save_figs_pdf('midterm_plt_figs.pdf')
