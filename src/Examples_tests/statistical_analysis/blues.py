import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import zscore
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt  
import math
from statsmodels.stats.multicomp import pairwise_tukeyhsd

df_blues = pd.read_csv("src/Examples_tests/Data/blues_score.csv")


# add groupe column to the dataframes
# Sample category list as a string
category_list = """
AA10MA Klassisch
BE01CL AR
BE13NA Klassisch
BE17MA AR
BI21FL Klassisch
BU01FR Klassisch
DA27SV Klassisch
DI03CA AR
FO09MA AR
GE03KA AR
GI16CA AR
JE13CL Klassisch
LU03GA Klassisch
SU13BA Klassisch
SU22PA AR
VA10SI Klassisch
WA24AN AR
ZU17AL AR
"""

# ----------- prepare data and clean --------------

# Convert to dictionary
category_dict = dict(line.split() for line in category_list.strip().split('\n'))

# Convert the dictionary to a DataFrame
df_blues['Category'] = df_blues[df_blues.columns[0]].map(category_dict)

# remove contestants that did not complete the study
df_blues = df_blues[df_blues['Category'].notna()]


# ------------ statistical analysis ----------------
# mean, std, min, max, count
df_stats = (df_blues.groupby('Category').describe())

# Group means
group_means = df_blues.groupby("Category").mean(numeric_only=True).T

#Correlation matrix
# Only Stück columns
stueck_cols = [col for col in df_blues.columns if "Blues" in col]

# Correlation matrix
corr = df_blues[stueck_cols].corr()


# -------------print / plot-----------------
print(df_stats)

# Plot the means
group_means.plot(kind="bar", figsize=(12, 6), title="Average per Stück by Category")
plt.ylabel("Average Value")
plt.xlabel("Stück")
plt.tight_layout()
plt.show()

# Plot the correlation matrix
# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Stück Variables")
plt.show()