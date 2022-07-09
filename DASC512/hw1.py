#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statistics as stat

'''Problem #3 '''
# list of responsible parties
lst_responsible = ['InsCos', 'PharmCos', 'Govt', 'Hosp', 'Phys', 'Other', 'Unsure']
  
# list of respondants
lst_nbr_responses = [869, 339, 338, 127, 85, 128, 23]
  
# function returns dict of rel freqs based on k,v pair
# params are in form of {string: int}
def compute_rel_freqs(dict:dict) -> dict:
    total_count = sum(dict.values())
    relative = {}
    for key in dict:
        relative[key] = dict[key] / total_count
    return relative

# create a dict of tuples representing the responsible parties
# and corresponsding numer of responses
dict_hc_cost = dict(zip(lst_responsible, lst_nbr_responses))

dict_rel_freqs = compute_rel_freqs(dict_hc_cost)

# bar plot of dict of rel_freqs
# sns_barplot = sns.barplot(x = list(dict_rel_freqs.keys()), y = list(dict_rel_freqs.values()))
# fig = sns_barplot.get_figure()
# fig.savefig("barplot_rel_freqs.png") 

''' Problem #4 '''
# Create a table of summary statistics for Spahn’s ERA. 
# This table should include the Min, Q1, Median, Mean, Q3, Max, Sample Variance and Sample Standard Deviation
df_spahn = pd.read_csv("spahn.csv", sep = ',')
# print(df_spahn.head(20))
# print(df_spahn['ERA'].describe())

# Spahn ERA min
spahn_era_min = np.min(df_spahn['ERA'])

# Spahn ERA Q1
spahn_era_q1 = np.quantile(df_spahn['ERA'], 0.25)

# Spahn ERA median
spahn_era_median = np.median(df_spahn['ERA'])

# Spahn ERA mean
spahn_era_mean = np.mean(df_spahn['ERA'])

spahn_era_q3 = np.quantile(df_spahn['ERA'], 0.75)

# Spahn ERA max
spahn_era_max = np.max(df_spahn['ERA'])

# Spahn ERA Sample Variance
spahn_era_svar = stat.variance(df_spahn['ERA'])

# Spahn ERA Sample Std Dev
spahn_era_std = stat.stdev(df_spahn['ERA']) 

# Spahn ERA Table
d = {'ERA': [spahn_era_min, spahn_era_q1, spahn_era_median, spahn_era_mean,
             spahn_era_q3, spahn_era_max, spahn_era_svar, spahn_era_std]}
# print(pd.DataFrame(data = d, index = ['min', 'q1', 'median', 'mean', 'q3', 'max', 'var', 'std']))

df_tm_era = pd.DataFrame.from_records(zip(df_spahn['Tm'] , df_spahn['ERA']), columns = ['Tm', 'ERA'])
df_bsn_mln = df_tm_era.loc[df_tm_era['Tm'] != 'TOT']
# print(df_bsn_mln)
bsn_mean = np.mean(df_bsn_mln.loc[df_bsn_mln['Tm'] == 'BSN']['ERA'])
bsn_median = np.median(df_bsn_mln.loc[df_bsn_mln['Tm'] == 'BSN']['ERA'])
spahn_bsn_svar = stat.variance(df_bsn_mln.loc[df_bsn_mln['Tm'] == 'BSN']['ERA'])
# print(bsn_mean)
# print(bsn_median)
# print(spahn_bsn_svar)
mln_mean = np.mean(df_bsn_mln.loc[df_bsn_mln['Tm'] == 'MLN']['ERA'])
mln_median = np.median(df_bsn_mln.loc[df_bsn_mln['Tm'] == 'MLN']['ERA'])
spahn_mln_svar = stat.variance(df_bsn_mln.loc[df_bsn_mln['Tm'] == 'MLN']['ERA'])
# print(mln_mean)
# print(mln_median)
# print(spahn_mln_svar)
# sns_boxplot = sns.boxplot(x = 'Tm', y = 'ERA', data = df_bsn_mln)
# fig = sns_boxplot.get_figure()
# fig.savefig("boxplot_spahn_bsn_mln.png")

'''Problem #5'''
# d5000.csv Create a scatterplot of the Home Runs versus the Strike Outs
df_d5000 = pd.read_csv("d5000.csv", sep = ',')
# print(df_d5000.head(20))
#Using Seaborn to Scatterplot
# sns.FacetGrid(df_d5000, hue='playerID', height=8).map(plt.scatter, 'HR', 'SO')
# plt.savefig("scatterplot_hr_so.png")

'''Problem #6'''
# Classification of eras is:
# “19th Century” (up to the 1900 Season), 
# “Dead Ball” (1901 through 1919), 
# “Lively Ball” (1920 through 1941), 
# “Integration” (1942 through 1960), 
# “Expansion” (1961 through 1976), 
# “Free Agency” (1977 through 1993), 
# “Long Ball” (after 1993). 
# Define the era a HoF as the era when he was at his mid-career 
# (the average of his first and last seasons in baseball), 
# use Python to divide the file ‘hofbatting.csv’ (containing all non-pitching HoF) 
# into the appropriate era subsets.
df_hof = pd.read_csv("hofbatting.csv", sep = ',')
# print(df_hof.head())

lst_mid_career_avg = df_hof[['From', 'To']].mean(axis=1).to_list()

lst_yr_mid_career = []
for i in lst_mid_career_avg:
    lst_yr_mid_career.append(int(round(i)))
# print(list_era)
df_yr_mid_career = pd.DataFrame(lst_yr_mid_career, columns = ['yr_mid_career'])
# print(df_yr_era)
df_hof_mid_career = df_hof.assign(yr_mid_career=df_yr_mid_career['yr_mid_career'])
# print(df_hof_mid_career.head(30))
# print('min_era => ', np.min(df_hof_mid_career['yr_mid_career']))

bins = [1800, 1900, 1919, 1941, 1960, 1976, 1993, 2013]
eras = ['19th Century', 'Dead Ball', 'Lively Ball', 'Integration', 
        'Expansion', 'Free Agency', 'Long Ball']
# print(df_hof_mid_career['yr_mid_career'].value_counts(bins=bins, sort=False))

# create a dict of tuples representing the eras, timeframes, counts
dict_eras = dict(zip(eras, df_hof_mid_career['yr_mid_career'].value_counts(bins=bins, sort=False)))
# print(dict_eras)

# ************Creating pie plot
fig = plt.figure(figsize =(4, 5))
values = df_hof_mid_career['yr_mid_career'].value_counts()
# plt.pie(dict_eras.values(), labels = dict_eras.keys(), autopct= lambda x: '{:.0f}'.format(x*values.sum()/100))
# plt.show()
# plt.savefig("pieplot_hof_eras.png")

# ************Creating Bar Plot
plt.bar(*zip(*dict_eras.items()), color = list('rgbkymc'))
plt.xticks(rotation='vertical')
plt.tight_layout()
# plt.show()
# plt.savefig("barplot_hof_eras.png")

# ************Creating Histogram
# Creating histogram
fig, ax = plt.subplots(1, 1,
                        figsize =(5, 3),
                        tight_layout = True)
plt.xlabel("Eras by Year")
plt.ylabel("Count")    
plt.title("Mid-career values for HoF non-pitchers")                    
hist_mid_career = ax.hist(df_hof_mid_career['yr_mid_career'], bins = bins)
plt.savefig("hist_hof_mid-career.png")
