#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

 
# list of responsible parties
lst_responsible = ['InsuranceCos', 'PharmaceuticalCos', 'Govt', 'Hospitals', 'Physicians', 'Other', 'Unsure']
  
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
sns_barplot = sns.barplot(x = list(dict_rel_freqs.keys()), y = list(dict_rel_freqs.values()))
fig = sns_barplot.get_figure()
fig.savefig("barplot_rel_freqs.png") 
