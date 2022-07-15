#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat

'''Problem 1'''
x = 2
x_minus_1 = x - 1
p5 = (0.23)*(0.77)**x_minus_1
# print(p5)
# print(1 - x.cdf(1)) # p(x ≥ 2)
# print(x.pmf(1)) # P(x = 1)
# print(x.pmf(5)) # P(x = 5)

'''Problem 2'''
x, p, n = 15, 0.8, 20
cum_prob = stats.binom.cdf(x, n, p)
print(cum_prob)
print(1 - cum_prob)
exact_prob = stats.binom.pmf(x, n, p)
print(exact_prob)
# obtaining the mean and variance 
mean, var = stats.binom.stats(n, p)
print(mean)

'''Problem 3'''
x, mu = 3, 5
prob_3 = stats.poisson.cdf(x, mu)
print(prob_3)

'''Problem 4'''
mean = 850,000
stdev = 170,000
R = 1,000,000

'''Problem 5'''
mu = 10