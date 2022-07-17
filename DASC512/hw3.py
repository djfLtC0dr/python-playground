'''Dan Fawcett DASC 512 Homework 3'''
#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat
from pathlib import Path

'''Problem 1a'''
placebo = (105, 119, 100, 97, 96, 101,94, 95, 98)
caffeine = (96, 99, 94, 89, 96, 93, 88, 105, 88)
mu = 100
alpha = 0.05
t_value, p_value = stats.ttest_1samp(placebo, mu)
# Since alternative hypothesis is one tailed, We need to divide the p value by 2.
one_tailed_p_value = float("{:.6f}".format(p_value/2)) 

print('Test statistic is %f'%float("{:.6f}".format(t_value)))
# print('p-value for one tailed test is %f'%one_tailed_p_value)

if one_tailed_p_value <= alpha:
    print('Conclusion: Since p-value(=%f)'%p_value,'<','alpha(=%.2f)'%alpha,'''We reject the null hypothesis H0. 
            So we conclude that there is no significant mean difference in RER 
            i.e., μ = 100 at %.2f level of significance'''%alpha)
else:
    print('Conclusion: Since p-value(=%f)'%one_tailed_p_value,'>','alpha(=%.2f)'%alpha,'' + '\n' +
    'We fail to reject  the null hypothesis H0.' + '\n' +
    'the mean value placebo NEN > 100')
