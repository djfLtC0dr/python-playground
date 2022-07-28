import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
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
df_faithful = pd.read_csv("faithful.csv", sep = ',')
# print(df_faithful.head())
eruptions = df_faithful['eruptions']
waiting = df_faithful['waiting']

plt.scatter(x=eruptions, y=waiting)
plt.show()