import pandas as pd
from matplotlib import pyplot as plt
import os

file_name = "Pipeline/parsed_classification_metrics_2.csv"
data = pd.read_csv(file_name)
languages = ['en', 'da', 'de', 'nl', 'sv', 'es', 'fr', 'it', 'pt', 'ro', 'bg', 'cs', 'hr', 'pl', 'sl', 'et', 'fi', 'hu', 'lt', 'lv', 'el']

output_folder = "multi_eurlex_plots"
os.makedirs(output_folder, exist_ok=True)

for language in languages:
    # Filter data for the current language
    predicted_label_distribution = data[data["Language"] == language].groupby("Category")["Predicted Num"].sum()
    true_label_distribution = data[data["Language"] == language].groupby("Category")["True Num"].sum()

    # Combine both distributions into a single DataFrame
    combined_df = pd.DataFrame({
        "Predicted": predicted_label_distribution,
        "True": true_label_distribution
    })

    # Plotting
    plt.figure(figsize=(12, 6))
    combined_df.plot(kind='bar', color=['skyblue', 'salmon'], width=0.8)
    plt.title(f"Distribution of True and Predicted Labels for {language}")
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.legend(["Predicted", "True"])
    plt.tight_layout()
    plt.show()

    plot_filename = os.path.join(output_folder, f"{language}_label_distribution.png")
    plt.savefig(plot_filename)
    plt.close()
