#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat
from scipy.stats import binom

'''Problem 1'''
# number of samples, probability of contamination
x = 2
x_minus_1 = x - 1
p5 = (0.23)*(0.77)**x_minus_1
# print(p5)
# print(1 - x.cdf(1)) # p(x ≥ 2)
# print(x.pmf(1)) # P(x = 1)
# print(x.pmf(5)) # P(x = 5)

x, p, n = 15, 0.8, 20
prob = stats.binom.cdf(x, n, p)
print(prob)
# np.allclose(x, stats.binom.ppf(prob, n, p))


