import pandas as pd
import numpy as np
from scipy.stats import t

df_finger = pd.read_csv("C:/Users/tobia/Desktop/Recording-comparison/src/Examples_tests/Data/fingergeschicklichkeit.csv")
df_sun = pd.read_csv("C:/Users/tobia/Desktop/Recording-comparison/src/Examples_tests/Data/risingsun_score.csv")
df_blues = pd.read_csv("C:/Users/tobia/Desktop/Recording-comparison/src/Examples_tests/Data/blues_score.csv")


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

# Convert to dictionary
category_dict = dict(line.split() for line in category_list.strip().split('\n'))

# Convert the dictionary to a DataFrame
df_finger['Category'] = df_finger[df_finger.columns[0]].map(category_dict)
df_sun['Category'] = df_sun[df_sun.columns[0]].map(category_dict)
df_blues['Category'] = df_blues[df_blues.columns[0]].map(category_dict)

# remove contestants that did not complete the study
df_finger = df_finger[df_finger['Category'].notna()]
df_sun = df_sun[df_sun['Category'].notna()]
df_blues = df_blues[df_blues['Category'].notna()]

##-----------analyze finger dexterity-------------------##
# separate the data into two groups based on the category
df_group1 = df_finger[df_finger['Category'] == 'Klassisch']
df_group2 = df_finger[df_finger['Category'] == 'AR']

# calculate the mean, standard deviation, standard error and confidence interval for each group

# ----- group 1 (klassisch) -----
df_numeric_group1 = df_group1.drop(columns=['Participant_ID', 'Category'])

# Sample size and degrees of freedom
n = len(df_numeric_group1)
dfree = n - 1

# Confidence level
confidence = 95
alpha = (100 - confidence) / 100  # e.g. 0.05 for 95% CI

# Calculate stats
mean = df_numeric_group1.mean()
std = df_numeric_group1.std() # Standard Deviation
se = std / np.sqrt(n) # Standard Error

# TINV equivalent: two-tailed t critical value
t_critical = t.ppf(1 - alpha / 2, dfree)

# Confidence interval bounds
ci_lower = mean - t_critical * se
ci_upper = mean + t_critical * se

# Combine into summary table
df_summary_group1 = pd.DataFrame({
    'Mean': mean,
    'Std Dev': std,
    'Std Error': se,
    'CI Lower (95%)': ci_lower,
    'CI Upper (95%)': ci_upper,
    'T Critical': t_critical
})


# ----- group 2 (AR) -----
df_numeric_group2 = df_group2.drop(columns=['Participant_ID', 'Category'])

# Sample size and degrees of freedom
n = len(df_numeric_group2)
dfree = n - 1

# Confidence level
confidence = 95
alpha = (100 - confidence) / 100  # e.g. 0.05 for 95% CI

# Calculate stats
mean = df_numeric_group2.mean()
std = df_numeric_group2.std() # Standard Deviation
se = std / np.sqrt(n) # Standard Error

# TINV equivalent: two-tailed t critical value
t_critical = t.ppf(1 - alpha / 2, dfree)

# Confidence interval bounds
ci_lower = mean - t_critical * se
ci_upper = mean + t_critical * se

# Combine into summary table
df_summary_group2 = pd.DataFrame({
    'Mean': mean,
    'Std Dev': std,
    'Std Error': se,
    'CI Lower (95%)': ci_lower,
    'CI Upper (95%)': ci_upper,
    'T Critical': t_critical
})


# ----check for statistical significance----
# Perform t-test
# Sample sizes
n1 = len(df_group1) 
n2 = len(df_group2)

# Store results
results = []

for col in df_summary_group1.index:
    mean1 = df_summary_group1.loc[col, 'Mean']
    mean2 = df_summary_group2.loc[col, 'Mean']
    sd1 = df_summary_group1.loc[col, 'Std Dev']
    sd2 = df_summary_group2.loc[col, 'Std Dev']

    # Welch's t-test calculation
    se_diff = np.sqrt((sd1**2 / n1) + (sd2**2 / n2))
    t_stat = (mean1 - mean2) / se_diff

    # Degrees of freedom (Welch-Satterthwaite equation)
    df_num = (sd1**2 / n1 + sd2**2 / n2)**2
    df_den = ((sd1**2 / n1)**2 / (n1 - 1)) + ((sd2**2 / n2)**2 / (n2 - 1))
    df_effective = df_num / df_den

    # Two-tailed p-value
    p_value = 2 * t.sf(np.abs(t_stat), df_effective)

    results.append({
        'Measure': col,
        'Mean group1': mean1,
        'Mean group2': mean2,
        't-statistic': t_stat,
        'df': df_effective,
        'p-value': p_value,
        'Significant (p < 0.05)': p_value < 0.05
    })

# Create results DataFrame
df_ttest_summary = pd.DataFrame(results)


print(df_finger)
print(df_summary_group1)
print(df_summary_group2)
print(df_ttest_summary)

# print(df_sun)
# print(df_blues)
