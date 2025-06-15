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



df_blues = pd.read_csv("src/data_analysis_pipeline/Data/blues_score2.csv")


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

df_blues_klassisch = df_blues[df_blues['Category'] == 'Klassisch']
print(df_blues_klassisch)


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

## spielen ar beim 3. termin signifikant besser als beim 1. termin?
# Nur Daten der AR-Gruppe auswählen
ar_data = df_blues[df_blues['Category'] == 'AR']
klassisch_data = df_blues[df_blues['Category'] == 'Klassisch']

# Termin 1 (HotRS_1-1) vs. Termin 3 (z. B. HotRS_3-2 oder 3-1?)
# Du musst entscheiden, ob du 3-1 oder 3-2 verwenden möchtest (z. B. 3-2 als "bessere" Nach-Messung)

t_stat, p_value = ttest_rel(ar_data['Blues_1-1'], ar_data['Blues_5-2'])
print("AR")
print(f"t({len(ar_data)-1}) = {t_stat:.2f}, p = {p_value:.4f}")

print("Klassisch")
t_stat_klassisch, p_value_klassisch = ttest_rel(klassisch_data['Blues_1-1'], klassisch_data['Blues_5-1'])
print(f"t({len(klassisch_data)-1}) = {t_stat_klassisch:.2f}, p = {p_value_klassisch:.4f}")

## test for significant differences between the two groups at all points
# --- Signifikanztests pro Termin: Klassisch vs. AR ---
results = []

for col in stueck_cols:
    data_klassisch = klassisch_data[col].dropna()
    data_ar = ar_data[col].dropna()

    # Normalverteilung prüfen
    stat_klass, p_klass = shapiro(data_klassisch)
    stat_ar, p_ar = shapiro(data_ar)

    normal_klass = p_klass > 0.05
    normal_ar = p_ar > 0.05

    if normal_klass and normal_ar:
        # t-Test, falls beide normalverteilt
        test_stat, p_val = ttest_ind(data_klassisch, data_ar, equal_var=False)
        test_type = "t-Test"
    else:
        # Mann-Whitney U-Test bei nicht-normalverteilten Daten
        test_stat, p_val = mannwhitneyu(data_klassisch, data_ar, alternative='two-sided')
        test_type = "Mann-Whitney-U"

    results.append({
        "Messzeitpunkt": col,
        "Testtyp": test_type,
        "p-Wert": round(p_val, 4),
        "Signifikant (p < 0.05)": "Ja" if p_val < 0.05 else "Nein"
    })

results_df = pd.DataFrame(results)
print(results_df)


# -------------print / plot-----------------
print(df_stats)

# # Plot the means
# group_means.plot(kind="bar", figsize=(12, 6), title="Average per Stück by Category")
# plt.ylabel("Average Value")
# plt.xlabel("Stück")
# plt.tight_layout()
# plt.show()

# # Plot the correlation matrix
# # Heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
# plt.title("Correlation Between Stück Variables")
# plt.show()





# Plot the means as line plot
# Mittelwert und Standardabweichung berechnen
group_summary = df_blues.groupby("Category")[stueck_cols].agg(['mean', 'std'])


# Definierte Reihenfolge und feine Positionen für zusammengehörige Punkte
stueck_cols = ['Blues_1-1', 'Blues_2-1', 'Blues_2-2', 'Blues_3-1', 'Blues_3-2', 'Blues_4-1', 'Blues_4-2', 'Blues_5-1', 'Blues_5-2']
x_positions = {
    'Blues_1-1': 1,
    'Blues_2-1': 2.0,
    'Blues_2-2': 2.15,
    'Blues_3-1': 3,
    'Blues_3-2': 3.15,
    'Blues_4-1': 4.0,
    'Blues_4-2': 4.15,
    'Blues_5-1': 5.0,
    'Blues_5-2': 5.15
}

# Plot vorbereiten
fig, ax = plt.subplots(figsize=(12, 6))
# X-Werte aus custom Positionen holen
x_vals = [x_positions[col] for col in stueck_cols]


for category in ["Klassisch", "AR"]:
    data = df_blues[df_blues['Category'] == category][stueck_cols]
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
ax.set_xlabel("Blues No. 1", fontsize=14)
ax.set_ylabel("Durchschnittliche Punktzahl", fontsize=14)
ax.set_title("Leistung nach Stück mit 95%-Konfidenzintervallen", fontsize=16)
ax.legend(title="Gruppe", loc='upper left', bbox_to_anchor=(0.1, 0.99), fontsize=14)
ax.grid(True)

# --- inset axes ---
inset_ax = inset_axes(ax, width="40%", height="40%", loc='upper right', borderpad=3)

for category in ["Klassisch", "AR"]:
    data = df_blues[df_blues['Category'] == category][stueck_cols]
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
    idxs_to_plot = [stueck_cols.index(col) for col in ['Blues_4-1', 'Blues_4-2', 'Blues_5-1', 'Blues_5-2']]
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
inset_ax.set_xticks([x_positions[col] for col in ['Blues_4-1', 'Blues_4-2', 'Blues_5-1', 'Blues_5-2']])
inset_ax.set_xticklabels(['Blues_4-1', 'Blues_4-2', 'Blues_5-1', 'Blues_5-2'], rotation=45)
inset_ax.set_ylim(bottom=-1000)  # Optional: adjust zoom level
inset_ax.set_title("Zoom: Blues No. 1 4 & 5", fontsize=14)
inset_ax.grid(True)

plt.tight_layout()
plt.show()