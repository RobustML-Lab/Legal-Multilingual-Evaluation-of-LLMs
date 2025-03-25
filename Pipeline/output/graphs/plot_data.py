import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("../../../Results/XNLI_Dataset_Performance_Data.csv", index_col=0)

# Define attack names
attack_names = {"14": "Contextual Word Embeddings", "11": "Character-level Modifications"}

# Rename index values for clarity
df.rename(index={
    "14": attack_names["14"],
    "11": attack_names["11"]
}, inplace=True)

languages = ["bg", "el", "en", "es", "fr", "th"]  # Include English
scenarios = ["Original", attack_names["14"], attack_names["11"]]

df_selected = df.loc[scenarios, languages].T

# ---- Plot Original, Attack 14, Attack 11 ----
fig, ax = plt.subplots(figsize=(10, 6))
df_selected.plot(kind="bar", ax=ax, width=0.8)
ax.set_title("Performance Comparison: Original vs. Attacks")
ax.set_ylabel("Score")
ax.set_xlabel("Languages")
ax.set_xticks(range(len(languages)))
ax.set_xticklabels(languages, rotation=45)  # Ensure proper x labels
ax.grid(axis="y")
ax.legend(title="Scenario", loc="upper right")  # Keep legend inside frame
plt.show()

# ---- Plot Original vs. Original Eng and 14 vs. 14 Eng (excluding English) ----
languages_no_eng = ["bg", "el", "es", "fr", "th"]  # Exclude English
df_eng_comp = df.loc[["Original", "Original Eng", attack_names["14"], "14 Eng"], languages_no_eng].T

fig, ax = plt.subplots(figsize=(10, 6))
df_eng_comp.plot(kind="bar", ax=ax, width=0.8)
ax.set_title("Comparison: Original vs. English Prompt")
ax.set_ylabel("Score")
ax.set_xlabel("Languages")
ax.set_xticks(range(len(languages_no_eng)))
ax.set_xticklabels(languages_no_eng, rotation=45)  # Ensure proper x labels
ax.grid(axis="y")
ax.legend(title="Scenario", loc="upper right")  # Keep legend inside frame
plt.show()

# ---- Compute Drop Percentage from Original in 14 and 11 ----
drop_percentage = ((df.loc["Original", languages] - df.loc[[attack_names["14"], attack_names["11"]], languages]) / df.loc["Original", languages]) * 100

fig, ax = plt.subplots(figsize=(10, 6))
drop_percentage.T.plot(kind="bar", ax=ax, width=0.8)
ax.set_title("Percentage Drop from Original Performance")
ax.set_ylabel("Drop (%)")
ax.set_xlabel("Languages")
ax.set_xticks(range(len(languages)))
ax.set_xticklabels(languages, rotation=45)  # Ensure proper x labels
ax.grid(axis="y")
ax.legend(title="Scenario", loc="upper right")  # Keep legend inside frame
plt.show()
