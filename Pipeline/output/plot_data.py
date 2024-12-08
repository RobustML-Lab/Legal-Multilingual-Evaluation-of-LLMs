import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
parsed_data = pd.read_csv('parsed_classification_metrics.csv')

# 1. Category-Level Performance by Language (Single Figure with Subplots)
languages = parsed_data['Language'].unique()
fig, axes = plt.subplots(len(languages), 3, figsize=(18, 5 * len(languages)))
fig.suptitle("Category-Level Performance by Language", fontsize=16)

for i, language in enumerate(languages):
    lang_data = parsed_data[parsed_data['Language'] == language]
    categories = lang_data['Category']

    # Precision plot
    sns.barplot(x='Precision', y=categories, data=lang_data, ax=axes[i, 0], palette='viridis')
    axes[i, 0].set_title(f"{language.upper()} - Precision")
    axes[i, 0].set_xlabel("Precision")
    axes[i, 0].set_ylabel("Category")

    # Recall plot
    sns.barplot(x='Recall', y=categories, data=lang_data, ax=axes[i, 1], palette='plasma')
    axes[i, 1].set_title(f"{language.upper()} - Recall")
    axes[i, 1].set_xlabel("Recall")
    axes[i, 1].set_ylabel("")

    # F1 Score plot
    sns.barplot(x='F1 Score', y=categories, data=lang_data, ax=axes[i, 2], palette='magma')
    axes[i, 2].set_title(f"{language.upper()} - F1 Score")
    axes[i, 2].set_xlabel("F1 Score")
    axes[i, 2].set_ylabel("")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 2. Global Metric Comparison Across Languages
global_metrics = parsed_data[['Language', 'Global Precision', 'Global Recall', 'Global F1 Score']].drop_duplicates()
global_metrics.set_index('Language', inplace=True)
global_metrics.plot(kind='bar', figsize=(12, 8))
plt.title("Global Precision, Recall, and F1 Score Across Languages")
plt.xlabel("Language")
plt.ylabel("Score")
plt.legend(title="Metrics")
plt.tight_layout()
plt.show()

# 3. Precision vs. Recall Scatter Plot by Category
plt.figure(figsize=(10, 12))
sns.scatterplot(x='Precision', y='Recall', hue='Language', data=parsed_data, style='Category', s=100)
plt.title("Precision vs Recall by Category")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(title="Language and Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 4. True vs Predicted Counts by Category with x=y line
plt.figure(figsize=(10, 12))
sns.scatterplot(x='True Num', y='Predicted Num', hue='Language', data=parsed_data, style='Category', s=100)
plt.plot([parsed_data['True Num'].min(), parsed_data['True Num'].max()],
         [parsed_data['True Num'].min(), parsed_data['True Num'].max()],
         color='gray', linestyle='--', linewidth=1)  # x=y line

plt.title("True vs Predicted Number of Instances by Category")
plt.xlabel("True Number of Instances")
plt.ylabel("Predicted Number of Instances")
plt.legend(title="Language and Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 5. True vs Predicted Counts by Category with x=y line (all languages combined)
mean_data = parsed_data.groupby('Category').agg({'True Num': 'mean', 'Predicted Num': 'mean'}).reset_index()

plt.figure(figsize=(10, 12))
sns.scatterplot(x='True Num', y='Predicted Num', hue='Category', data=mean_data, s=100, alpha=0.7)
plt.plot([mean_data['True Num'].min(), mean_data['True Num'].max()],
         [mean_data['True Num'].min(), mean_data['True Num'].max()],
         color='gray', linestyle='--', linewidth=1)  # x=y line

plt.title("Average True vs Predicted Number of Instances by Category (Across All Languages)")
plt.xlabel("Average True Number of Instances")
plt.ylabel("Average Predicted Number of Instances")
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()