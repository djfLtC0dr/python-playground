#Standard Import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import csv

#Load some data from CSV file to play around with
def read_dataset_csv_file(file_path: str) -> list:
    with open(file_path) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        lst_tuples = []
        for row in csvReader:
            lst_tuples.append(row)
        return tuple(lst_tuples)

#spahn_data = read_dataset_csv_file("DASC512/spahn.csv")

df_spahn = pd.read_csv("DASC512/spahn.csv", sep = ',')
print(df_spahn.head())
# print(spahn_data)
