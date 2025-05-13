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


df_finger = pd.read_csv("src/Examples_tests/Data/fingergeschicklichkeit.csv")


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

#Convert to dictionary
category_dict = dict(line.split() for line in category_list.strip().split('\n'))

# Convert the dictionary to a DataFrame
df_finger['Category'] = df_finger[df_finger.columns[0]].map(category_dict)

# remove contestants that did not complete the study
df_finger = df_finger[df_finger['Category'].notna()]

# JE13CL is a special case, played wrong sequence therefore just take the his played sequence calculated in saving_JE13CL.py
Finger_1_1_correct = 16
Finger_1_2_correct = 26

df_finger.loc[df_finger['Participant_ID'] == 'JE13CL', 'Finger_1-1_correct'] = Finger_1_1_correct
df_finger.loc[df_finger['Participant_ID'] == 'JE13CL', 'Finger_1-2_correct'] = Finger_1_2_correct

# Anovatest

df_finger_clean = df_finger.rename(columns=lambda x: x.replace('-', '_'))

# List of columns to test
cols_to_test = [col for col in df_finger_clean.columns if ('correct' in col or 'keys' in col)]

# Run ANOVA for each and collect significant columns
significant_cols = []
for col in cols_to_test:
    model = smf.ols(f'{col} ~ C(Category)', data=df_finger_clean).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    p_value = anova['PR(>F)'][0]
    print(f"ANOVA for {col}: p = {p_value:.4f}")
    if p_value < 0.05:
        print("  → Significant! Adding to Tukey HSD plots...")
        significant_cols.append(col)
    else:
        print("  → Not significant, skipping Tukey.")


# Plot all Tukey HSD results in one figure
if significant_cols:
    fig, axes = plt.subplots(nrows=len(significant_cols), figsize=(8, 4 * len(significant_cols)))
    if len(significant_cols) == 1:
        axes = [axes]  # make it iterable
    for ax, col in zip(axes, significant_cols):
        tukey = pairwise_tukeyhsd(endog=df_finger_clean[col], groups=df_finger_clean['Category'], alpha=0.05)
        tukey.plot_simultaneous(ax=ax)
        ax.set_title(f'Tukey HSD: {col}')
    plt.tight_layout()
    plt.show()

# --------- plot the results -----------
# Select columns to plot
cols = [c for c in df_finger_clean.columns if 'correct' in c or 'keys' in c]
num_plots = len(cols)

# Set grid size
cols_per_row = 4  # Adjust this to control layout
rows = math.ceil(num_plots / cols_per_row)

# Create subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(6 * cols_per_row, 5 * rows))
axes = axes.flatten()  # Flatten in case of multi-row layout

# Plot each boxplot
for i, col in enumerate(cols):
    ax = axes[i]
    sns.boxplot(x='Category', y=col, data=df_finger_clean, ax=ax)
    sns.stripplot(x='Category', y=col, data=df_finger_clean, color='black', size=4, jitter=True, ax=ax)
    ax.set_title(f'{col} by Category')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    
    # Set y-axis limits
    if 'correct' in col:
        ax.set_ylim(5, 30)
    elif 'keys' in col:
        ax.set_ylim(60, 180)

# Remove empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()