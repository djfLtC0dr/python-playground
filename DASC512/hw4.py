'''Dan Fawcett DASC 512 Homework 4'''
#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import csv

'''Problem 1'''
# A study was designed to determine if a persons essay quality or attractiveness 
# played a role in the grading of essays. Column 1 in the data is essay quality 
# (rated good=1,poor=2), the second column is student attractiveness (1=attractive, 2=control, 3=unattractive). And the third column is the response (essay score).
# Construct an interaction plot and the full anova table (row for treatment as well as each factor). The .dat file is a tab delimited text file

# Import data file into a dataframe
txt_file = r"halo1.dat"
# essay quality = eq, student attractiveness = sa, essay score = score
COLUMN_NAMES=['eq','sa','score']

data = pd.DataFrame()
with open(txt_file) as f:
    data = pd.read_table(f, delim_whitespace=True, header=None, names=COLUMN_NAMES,
                          lineterminator='\n')
# print(data)
#Create Interaction Plot
fig, ax = plt.subplots(figsize=(12,6))
fig = interaction_plot(data['eq'], data['sa'], data['score'],
                       colors=['green', 'yellow', 'blue'], markers=['1', '2', '3'], ms=10, ax=ax)
#Calculate Degrees of Freedom
n=len(data.score)
df_a = len(data['eq'].unique())-1
df_b = len(data['sa'].unique())-1
df_ab = df_a*df_b
df_e = n-len(data['eq'].unique())*len(data['sa'].unique())
df_t = df_a+df_b+df_ab

#Calculate Sum of Squares
grand_mean = data['score'].mean()

ssa = sum([(data[data['eq'] ==l].score.mean()-grand_mean)**2 for l in data['eq']])

ssb = sum([(data[data['sa'] ==l].score.mean()-grand_mean)**2 for l in data['sa']])

sstotal = sum((data['score'] - grand_mean)**2)

eq1 = data[data['eq'] == 1]
eq2 = data[data['eq'] == 2]
eq1_sa_means = [eq1[eq1['sa'] == a].score.mean() for a in eq1['sa']]
eq2_sa_means = [eq2[eq2['sa'] == a].score.mean() for a in eq2['sa']]
sse = sum((eq1.score - eq1_sa_means)**2) +sum((eq2.score - eq2_sa_means)**2)

ssab = sstotal-ssa-ssb-sse                                         

sst = ssab+ssa+ssb

#Calculate Mean Squares
msa = ssa/df_a
msb = ssb/df_b
msab = ssab/df_ab
mse = sse/df_e
mst = sst/df_t

#F Ratios
f_a = msa/mse
f_b = msb/mse
f_ab = msab/mse

#P values
p_a = stats.f.sf(f_a, df_a, df_e)
p_b = stats.f.sf(f_b, df_b, df_e)
p_ab = stats.f.sf(f_ab, df_ab, df_e)

results = {'sum_sq':[ssa, ssb, ssab, sse],
           'df':[df_a, df_b, df_ab, df_e],
           'MS':[msa, msb, msab, mse],
           'F':[f_a, f_b, f_ab, 'NaN'],
            'PR(>F)':[p_a, p_b, p_ab, 'NaN']}
columns=['sum_sq', 'df', 'MS', 'F', 'PR(>F)']
aov_table1 = pd.DataFrame(results, columns=columns,
                          index=['eq', 'sa', 
                          'eq:sa', 'Residual'])

print(aov_table1)


formula = 'score ~ C(eq) + C(sa) + C(eq):C(sa)'
model = ols(formula, data).fit()
aov_table = anova_lm(model, typ=2)

print(aov_table)