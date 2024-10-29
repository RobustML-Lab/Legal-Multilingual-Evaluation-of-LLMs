import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

# Load the processed data
data = pd.read_csv("parsed_classification_metrics_2.csv")

# Set seaborn theme for better visuals
sns.set_theme(style="whitegrid")

# 1. Precision, Recall, and F1 Score by Category across Languages
plt.figure(figsize=(15, 10))
sns.boxplot(x='Category', y='F1 Score', data=data, palette="Set3")
plt.xticks(rotation=90)
plt.title("F1 Score by Category Across Languages")
plt.xlabel("Category")
plt.ylabel("F1 Score")
plt.tight_layout()
plt.show()

# 1. Precision, Recall, and F1 Score by Category across Languages
plt.figure(figsize=(15, 10))
sns.boxplot(x='Category', y='Precision', data=data, palette="Set3")
plt.xticks(rotation=90)
plt.title("Precision Score by Category Across Languages")
plt.xlabel("Category")
plt.ylabel("Precision")
plt.tight_layout()
plt.show()

# 1. Precision, Recall, and F1 Score by Category across Languages
plt.figure(figsize=(15, 10))
sns.boxplot(x='Category', y='Recall', data=data, palette="Set3")
plt.xticks(rotation=90)
plt.title("Recall Score by Category Across Languages")
plt.xlabel("Category")
plt.ylabel("Recall")
plt.tight_layout()
plt.show()

# 2. Per-Language Performance Analysis
languages = data['Language'].unique()
num_languages = len(languages)

# Determine appropriate grid size for subplots (e.g., 3 columns)
cols = 3
rows = math.ceil(num_languages / cols)

fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
axes = axes.flatten()  # Flatten to easily iterate through axes

for idx, lang in enumerate(languages):
    ax = axes[idx]
    subset = data[data['Language'] == lang]
    sns.barplot(x='Category', y='F1 Score', data=subset, ax=ax, hue='Category', dodge=False)
    ax.set_title(f'Performance in {lang}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_ylim(0, 1)

# Remove any extra subplots that were not used
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# 3. Correlation Between Num and Performance Metrics
# Define colors for each language
languages = data['Language'].unique()
palette = sns.color_palette("hsv", len(languages))
language_color_mapping = {lang: palette[idx] for idx, lang in enumerate(languages)}

# Plot Correlation Between Num and Performance Metrics
plt.figure(figsize=(15, 10))

# Plot each metric with a.py different marker
scatter1 = sns.scatterplot(x='True Num', y='Precision', hue='Language', data=data, palette=language_color_mapping, s=100, marker='o', legend=False)
scatter2 = sns.scatterplot(x='True Num', y='Recall', hue='Language', data=data, palette=language_color_mapping, s=100, marker='x', legend=False)
scatter3 = sns.scatterplot(x='True Num', y='F1 Score', hue='Language', data=data, palette=language_color_mapping, s=100, marker='v', legend=False)

plt.xscale('log')
plt.title("Correlation Between `Num` (Rarity) and Performance Metrics")
plt.xlabel("Num (log scale)")
plt.ylabel("Performance Scores")

# Legend for languages (colors)
from matplotlib.patches import Patch
language_legend_elements = [Patch(facecolor=color, edgecolor='black', label=lang) for lang, color in language_color_mapping.items()]
legend1 = plt.legend(handles=language_legend_elements, title="Languages", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Separate legend for symbols (metrics)
from matplotlib.lines import Line2D
metric_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Precision'),
    Line2D([0], [0], marker='x', color='w', markerfacecolor='gray', markersize=10, label='Recall'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=10, label='F1 Score')
]
legend2 = plt.legend(handles=metric_legend_elements, title="Metrics", bbox_to_anchor=(1.05, 0.7), loc='upper left', borderaxespad=0.)

# Add the language legend back so both appear
plt.gca().add_artist(legend1)

plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout

plt.show()

# 4. Distribution of Scores Across All Categories
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.boxplot(x='Precision', data=data, ax=axes[0])
sns.boxplot(x='Recall', data=data, ax=axes[1])
sns.boxplot(x='F1 Score', data=data, ax=axes[2])

axes[0].set_title("Precision Distribution")
axes[1].set_title("Recall Distribution")
axes[2].set_title("F1 Score Distribution")
for ax in axes:
    ax.set_xlim(0, 1)

plt.tight_layout()
plt.show()



# Aggregate the data by category, summing up `True Num` and `Predicted Num`
aggregated_data = data.groupby("Category", as_index=False).agg({
    "True Num": "sum",
    "Predicted Num": "sum"
})

# Scatter Plot: Aggregated `True Num` vs. `Predicted Num` Across All Categories
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x='True Num',
    y='Predicted Num',
    hue='Category',
    data=aggregated_data,
    palette="tab20",  # Using a.py palette with more colors for better distinction
    s=100,
    alpha=0.7
)
plt.xscale('log')
plt.yscale('log')
plt.plot([1, aggregated_data['True Num'].max()], [1, aggregated_data['True Num'].max()], color='red', linestyle='--')
plt.title("Aggregated Analysis: `True Num` vs. `Predicted Num` Across All Categories")
plt.xlabel("True Num (log scale)")
plt.ylabel("Predicted Num (log scale)")

# Place the legend outside the graph to avoid clutter
plt.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()