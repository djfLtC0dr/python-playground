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

# A study was designed to determine if a persons essay quality or attractiveness 
# played a role in the grading of essays. Column 1 in the data is essay quality 
# (rated good=1,poor=2), the second column is student attractiveness (1=attractive, 2=control, 3=unattractive). And the third column is the response (essay score).
# Construct an interaction plot and the full anova table (row for treatment as well as each factor). The .dat file is a tab delimited text file

# Refactor this into a dataframe
txt_file = r"halo1.dat"
COLUMN_NAMES=['quality','attractiveness','response']

df_essays = pd.DataFrame()
with open(txt_file) as f:
    df_essays = pd.read_table(f, delim_whitespace=True, header=None, names=COLUMN_NAMES,
                          lineterminator='\n')
# print(df_essays)
