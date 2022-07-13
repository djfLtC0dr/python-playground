#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat

'''Problem 1'''
# number of samples, probability of contamination
n = 2 # n=1 for P(x = 1) n=5 for P(x = 5)
p = .23 
x = stats.binom(n, p)
print(1 - x.cdf(1)) # p(x ≥ 2)
# print(x.pmf(1)) # P(x = 1)
# print(x.pmf(5)) # P(x = 5)



