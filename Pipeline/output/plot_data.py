import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_and_plot_drop(file_google4, file_google_before):
    # Load the data from the files
    google4_data = pd.read_csv(file_google4, sep=":", header=None, names=["Metric", "Value"], skip_blank_lines=True)
    google_before_data = pd.read_csv(file_google_before, sep=":", header=None, names=["Metric", "Value"], skip_blank_lines=True)

    # Preprocess the data
    def extract_results(data):
        results = {}
        current_language = None
        for _, row in data.iterrows():
            if row["Metric"].startswith("Results for"):
                current_language = row["Metric"].split("Results for ")[1].strip().lower()
                results[current_language] = {}
            elif current_language:
                metric_name, value = row["Metric"].strip(), row["Value"]
                if metric_name and value:
                    results[current_language][metric_name] = float(value)
        return pd.DataFrame(results).T

    df_google4 = extract_results(google4_data)
    df_google_before = extract_results(google_before_data)

    # Merge the data based on languages
    merged_df = pd.merge(df_google4, df_google_before, left_index=True, right_index=True, suffixes=("_google4", "_before"))

    # Calculate the percentage drop for each metric
    metrics = ["Rouge1", "Rouge2", "RougeL", "RougeL sum"]
    for metric in metrics:
        merged_df[f"{metric}_drop_percentage"] = (
                (merged_df[f"{metric}_before"] - merged_df[f"{metric}_google4"]) / merged_df[f"{metric}_before"] * 100
        )

    # Separate English and non-English for graph
    non_english_df = merged_df.drop(index="english", errors='ignore')
    english_df = merged_df.loc["english"] if "english" in merged_df.index else pd.DataFrame()

    # Calculate average drop percentages for English and non-English
    average_drops = {
        "Metric": metrics,
        "English": [english_df[f"{metric}_drop_percentage"].mean() if not english_df.empty else 0 for metric in metrics],
        "Non-English": [non_english_df[f"{metric}_drop_percentage"].mean() for metric in metrics],
    }

    # Create a DataFrame for the average drops
    average_drops_df = pd.DataFrame(average_drops)

    # Bar plot for average drop percentages
    x = np.arange(len(metrics))  # Number of metrics
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, average_drops_df["English"], width, label="English", color="red")
    bars2 = ax.bar(x + width / 2, average_drops_df["Non-English"], width, label="Non-English", color="blue")

    # Add labels and formatting
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Average Drop Percentage")
    ax.set_title("Average Drop Percentage by Metric (English vs Non-English)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def calculate_average_results(file_google4, file_google_before):
    # Load the data from the files
    google4_data = pd.read_csv(file_google4, sep=":", header=None, names=["Metric", "Value"], skip_blank_lines=True)
    google_before_data = pd.read_csv(file_google_before, sep=":", header=None, names=["Metric", "Value"], skip_blank_lines=True)

    # Preprocess the data
    def extract_results(data):
        results = {}
        current_language = None
        for _, row in data.iterrows():
            if row["Metric"].startswith("Results for"):
                current_language = row["Metric"].split("Results for ")[1].strip().lower()
                results[current_language] = {}
            elif current_language:
                metric_name, value = row["Metric"].strip(), row["Value"]
                if metric_name and value:
                    results[current_language][metric_name] = float(value)
        return pd.DataFrame(results).T

    df_google4 = extract_results(google4_data)
    df_google_before = extract_results(google_before_data)

    # Separate English and non-English
    non_english_google4 = df_google4.drop(index="english", errors='ignore')
    english_google4 = df_google4.loc[["english"]] if "english" in df_google4.index else pd.DataFrame()

    non_english_before = df_google_before.drop(index="english", errors='ignore')
    english_before = df_google_before.loc[["english"]] if "english" in df_google_before.index else pd.DataFrame()

    # Calculate average results for each metric
    metrics = ["Rouge1", "Rouge2", "RougeL", "RougeL sum"]
    average_results_google4 = {
        "Metric": metrics,
        "English": [english_google4[metric].mean() if not english_google4.empty else 0 for metric in metrics],
        "Non-English": [non_english_google4[metric].mean() for metric in metrics],
    }

    average_results_before = {
        "Metric": metrics,
        "English": [english_before[metric].mean() if not english_before.empty else 0 for metric in metrics],
        "Non-English": [non_english_before[metric].mean() for metric in metrics],
    }

    avg_results_google4_df = pd.DataFrame(average_results_google4)
    avg_results_before_df = pd.DataFrame(average_results_before)

    # Plot results for Google4 (with attack) and Before (without attack)
    x = np.arange(len(metrics))  # Number of metrics
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, avg_results_before_df["Non-English"], width, label="Non-English (Without Attack)", color="blue")
    bars2 = ax.bar(x + width / 2, avg_results_google4_df["Non-English"], width, label="Non-English (With Attack)", color="orange")

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Average Scores")
    ax.set_title("Average Results for Non-English (With vs Without Attack)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, avg_results_before_df["English"], width, label="English (Without Attack)", color="red")
    bars2 = ax.bar(x + width / 2, avg_results_google4_df["English"], width, label="English (With Attack)", color="green")

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Average Scores")
    ax.set_title("Average Results for English (With vs Without Attack)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot Google4 results (With attack) for English and Non-English
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, avg_results_google4_df["Non-English"], width, label="Non-English", color="blue")
    bars2 = ax.bar(x + width / 2, avg_results_google4_df["English"], width, label="English", color="red")

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Average Scores")
    ax.set_title("Google4 Results (With Attack) for English and Non-English")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Example usage
process_and_plot_drop("attacks/google_prompt1.log", "attacks/google_before.log")
calculate_average_results("attacks/google_prompt1.log", "attacks/google_before.log")
