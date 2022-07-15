#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat

'''Problem 1'''
x = 1
x_minus_1 = x - 1
p = (0.23)*(0.77)**x_minus_1
print(p)
p_2 = 1 - p
print(p_2)

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
u = 850000 # mean
sigma = 170000 #stdev
r = 1000000 #revenue

profit_dist =  stats.norm(u, sigma)

# CDF = the integral of the PDF, i.e. the area under the curve created by the PDF
# This is the probability of earning a profit > 0
prob_profit = 1 - profit_dist.cdf(0)
print(prob_profit)

# prob of loss
prob_loss = profit_dist.cdf(0)
print(prob_loss)

# prob of turning profit 99%
r =  stats.norm.ppf(.99, u, sigma)
#profit_dist.pdf(.99)
print(r)


'''Problem 5'''
from pathlib import Path
# function to check if a file exists 
# args file name assumes in root of running dir
def bool_file_exists(file:str) -> bool:
    filesystem_path = Path(file)
    return filesystem_path.is_file()

big_n = 100 # Nbr samples
λ = 10

poisson_prob_dist = stats.poisson.pmf(range(big_n), λ)

# Uniform dist observations
np.random.seed(1)
n = 2 # Nbr observations each sample

means = [np.mean(stats.uniform.rvs(size=n)) for i in poisson_prob_dist]
fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, 
            gridspec_kw={"height_ratios": (.15, .85)}, figsize = (12, 8))
fig.tight_layout(pad = 3)
fig.suptitle('Uniform CLT Boxplot & Histogram')
sns.boxplot(data=means, ax=ax_box, color = 'darkgreen')
sns.histplot(data=means, ax=ax_hist, color = 'green', kde=True, stat="density", linewidth=0)
boxdistplots_n10_bign100_file = "boxdistplots_n10_bign100.png"
if bool_file_exists(boxdistplots_n10_bign100_file) == False:
    fig.savefig(boxdistplots_n10_bign100_file) 