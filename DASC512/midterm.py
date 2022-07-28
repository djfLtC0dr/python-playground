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
plt.scatter(x=eruptions, y=waiting)
plt.title("Eruptions vs Wait Times for Ol' Faithful")
plt.xlabel("Eruptions)")
plt.ylabel("Wait Times")
# plt.show()

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
            (i.e. accept H_a) and conclude mean wait time avg < 60" 
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