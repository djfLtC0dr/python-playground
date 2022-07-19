'''Dan Fawcett DASC 512 Homework 3'''
#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat
from pathlib import Path

'''Problem 1a Right-Tail Test'''
placebo = (105, 119, 100, 97, 96, 101, 94, 95, 98)
mu = 100
alpha = 0.05
t_stat, p_val = stats.ttest_1samp(placebo, mu)
# Since alternative hypothesis is one tailed, We need to divide the p value by 2.
p_val_upper_tail = float("{:.6f}".format(p_val/2)) 
# usually dof = n - 1 for a single population sampling problem
dof = len(placebo) - 1
t_crit = stats.t.ppf(1 - alpha, dof)

print('Test statistic is %f'%float("{:.6f}".format(t_stat)))
print('p-value for right-tailed test is %f'%p_val_upper_tail)
print('T-Crit for this right-tailed placebo test is %f'%float("{:.6f}".format(t_crit)))

if t_stat <= t_crit:
    print('Since t_stat(=%f)'%t_stat,'<=','t_crit(=%.2f)'%t_crit,'''We reject the null hypothesis H0. 
            So we conclude that there is no significant mean difference in RER 
            i.e., μ = 100 at %.2f level of significance'''%alpha)
else:
    print('Since t_stat(=%f)'%t_stat,'>','t_crit(=%.2f)'%t_crit,'' + '\n' +
    'We fail to reject  the null hypothesis H0.' + '\n' +
    'we conclude the mean value placebo NEN > 100')

'''Problem 1b Left-Tail Test'''
caffeine = (96, 99, 94, 89, 96, 93, 88, 105, 88)
mu = 92
t_stat, p_val = stats.ttest_1samp(caffeine, mu)
# p_value is wrong, it's for 2-sided, which is equivalent to Upper tail/2.  
# To convert subtract 1 and divide by 2
p_val_lower_tail= float("{:.6f}".format((1 - p_val) / 2))
# usually dof = n - 1 for a single population sampling problem
dof = len(caffeine) - 1
t_crit = -stats.t.ppf(alpha, dof)
print(t_stat, p_val, p_val_lower_tail, t_crit)

if p_val_lower_tail <= alpha:
    print('Conclusion: Since t_stat(=%f)'%t_stat,'<','t_crit(=%.2f)'%t_crit,'''We reject the null hypothesis H0. 
            So we conclude that there is no significant mean difference in RER 
            i.e., μ = 92 at %.2f level of significance'''%alpha)
else:
    print('Since t_stat(=%f)'%p_val_upper_tail,'>','t_crit(=%.2f)'%t_crit,'' + '\n' +
    'We fail to reject the null hypothesis H0.' + '\n' +
    'the mean value coffee NEN < 92')

'''Problem 1c diff in Mean Values'''
print(np.std(placebo, ddof=1), np.std(caffeine, ddof=1))
t_stat, p_val=stats.ttest_ind(placebo, caffeine, equal_var=False)
t_crit=stats.t.ppf((1-alpha)/2,16)
print(t_stat, p_val, t_crit)
#Fail to reject the null, these means are statistically equal
