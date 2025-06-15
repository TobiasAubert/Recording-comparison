import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import zscore
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt  
import math
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import shapiro


df_sun = pd.read_csv("src/data_analysis_pipeline/Data/risingsun_score2.csv")
df_sun.rename(columns={col: col.replace("Stück", "HotRS") for col in df_sun.columns if col.startswith("Stück")}, inplace=True)
print(df_sun.head())


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
df_sun['Category'] = df_sun[df_sun.columns[0]].map(category_dict)

# remove contestants that did not complete the study
df_sun = df_sun[df_sun['Category'].notna()]


# ------------ statistical analysis ----------------
# mean, std, min, max, count
df_stats = (df_sun.groupby('Category').describe())

# Group means
group_means = df_sun.groupby("Category").mean(numeric_only=True).T

#Correlation matrix
# Only HotRS columns
stueck_cols = [col for col in df_sun.columns if "HotRS" in col]

# Correlation matrix
corr = df_sun[stueck_cols].corr()

### t-tests 

## spielen ar beim 3. termin signifikant besser als beim 1. termin?
# Nur Daten der AR-Gruppe auswählen
ar_data = df_sun[df_sun['Category'] == 'AR']

# Termin 1 (HotRS_1-1) vs. Termin 3 (z. B. HotRS_3-2 oder 3-1?)
# Du musst entscheiden, ob du 3-1 oder 3-2 verwenden möchtest (z. B. 3-2 als "bessere" Nach-Messung)

t_stat, p_value = ttest_rel(ar_data['HotRS_1-1'], ar_data['HotRS_3-2'])

print(f"t({len(ar_data)-1}) = {t_stat:.2f}, p = {p_value:.4f}")

## signifikanten Unterschied zwischen den Gruppen bei HotRS_5_2 beide gruppen?

# Daten extrahieren
klassisch = df_sun[df_sun['Category'] == 'Klassisch']['HotRS_5-2']
ar = df_sun[df_sun['Category'] == 'AR']['HotRS_5-2']

# Normalverteilung testen
shapiro_k = shapiro(klassisch)
shapiro_a = shapiro(ar)

print(f"Shapiro Klassisch: p = {shapiro_k.pvalue:.4f}")
print(f"Shapiro AR: p = {shapiro_a.pvalue:.4f}")

# Test wählen je nach Verteilung
if shapiro_k.pvalue > 0.05 and shapiro_a.pvalue > 0.05:
    # t-Test, wenn beide normalverteilt
    t_stat, p_val = ttest_ind(klassisch, ar, equal_var=False)
    print(f"t-Test: t = {t_stat:.2f}, p = {p_val:.4f}")
else:
    # Mann-Whitney-U-Test, wenn mind. eine Gruppe nicht normalverteilt
    u_stat, p_val = mannwhitneyu(klassisch, ar, alternative='two-sided')
    print(f"Mann-Whitney-U-Test: U = {u_stat:.2f}, p = {p_val:.4f}")


# -------------print / plot-----------------
print(df_stats)

# # Plot the means
# group_means.plot(kind="bar", figsize=(12, 6), title="Average per HotRS by Category")
# plt.ylabel("Average Value")
# plt.xlabel("HotRS")
# plt.tight_layout()
# plt.show()

# # Plot the correlation matrix
# # Heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Between HotRS Variables")
# plt.show()



# Plot the means as line plot
# Mittelwert und Standardabweichung berechnen
group_summary = df_sun.groupby("Category")[stueck_cols].agg(['mean', 'std'])


# Definierte Reihenfolge und feine Positionen für zusammengehörige Punkte
stueck_cols = ['HotRS_1-1', 'HotRS_2-1', 'HotRS_2-2', 'HotRS_3-1', 'HotRS_3-2', 'HotRS_4-1', 'HotRS_4-2', 'HotRS_5-1', 'HotRS_5-2']
x_positions = {
    'HotRS_1-1': 1,
    'HotRS_2-1': 2.0,
    'HotRS_2-2': 2.15,
    'HotRS_3-1': 3,
    'HotRS_3-2': 3.15,
    'HotRS_4-1': 4.0,
    'HotRS_4-2': 4.15,
    'HotRS_5-1': 5.0,
    'HotRS_5-2': 5.15
}

# Plot vorbereiten

# Plot vorbereiten
fig, ax = plt.subplots(figsize=(12, 6))
# X-Werte aus custom Positionen holen
x_vals = [x_positions[col] for col in stueck_cols]


for category in ["Klassisch", "AR"]:
    data = df_sun[df_sun['Category'] == category][stueck_cols]
    means = data.mean()
    stds = data.std()
    n = len(data)

    t_value = t.ppf(1 - 0.025, df=n - 1)
    ci = t_value * stds / np.sqrt(n)

    if category == "AR":
        jittered_x = [x - 0.05 for x in x_vals]
    else:
        jittered_x = x_vals

    ax.errorbar(
        jittered_x,
        means,
        yerr=ci,
        label=category,
        marker='o',
        capsize=4,
        linestyle='-'
    )

ax.set_xticks(x_vals)
ax.set_xticklabels(stueck_cols, rotation=45)
ax.set_xlabel("House of the Rising Sun Stück (HotRS)", fontsize=14)
ax.set_ylabel("Durchschnittliche Punktzahl", fontsize=14)
ax.set_title("Leistung nach Stück mit 95%-Konfidenzintervallen", fontsize=16)
ax.legend(title="Gruppe", loc='upper left', bbox_to_anchor=(0.1, 0.99), fontsize=14)
ax.grid(True)

# --- inset axes ---
inset_ax = inset_axes(ax, width="40%", height="40%", bbox_to_anchor=(-0.01, -0.05, 1, 1), bbox_transform=ax.transAxes)

for category in ["Klassisch", "AR"]:
    data = df_sun[df_sun['Category'] == category][stueck_cols]
    means = data.mean()
    stds = data.std()
    n = len(data)
    t_value = t.ppf(1 - 0.025, df=n - 1)
    ci = t_value * stds / np.sqrt(n)

    if category == "AR":
        jittered_x = [x_positions[col] - 0.05 for col in stueck_cols]
    else:
        jittered_x = [x_positions[col] for col in stueck_cols]

    # Only plot 4_x and 5_x values
    idxs_to_plot = [stueck_cols.index(col) for col in ['HotRS_4-1', 'HotRS_4-2', 'HotRS_5-1', 'HotRS_5-2']]
    inset_ax.errorbar(
        [jittered_x[i] for i in idxs_to_plot],
        means.iloc[idxs_to_plot],
        yerr=ci.iloc[idxs_to_plot],
        label=category,
        marker='o',
        capsize=3,
        linestyle='-'
    )

# Customize inset axes
inset_ax.set_xticks([x_positions[col] for col in ['HotRS_4-1', 'HotRS_4-2', 'HotRS_5-1', 'HotRS_5-2']])
inset_ax.set_xticklabels(['HotRS_4-1', 'HotRS_4-2', 'HotRS_5-1', 'HotRS_5-2'], rotation=45)
inset_ax.set_ylim(bottom=-1000)  # Optional: adjust zoom level
inset_ax.set_title("Zoom: House of the Rising Sun 4 & 5", fontsize=14)
inset_ax.grid(True)

plt.tight_layout()
plt.show()