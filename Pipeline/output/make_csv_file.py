import re
import pandas as pd

def parse_file_with_markers(content):
    # Patterns to identify key components based on markers
    language_pattern = re.compile(r'Results for (\w+):')
    global_metrics_pattern = re.compile(r'Precision: ([\d.]+)\s+Recall: ([\d.]+)\s+F1 Score: ([\d.]+)\s+Length: (\d+)\s+ENDMETRICS')
    class_pattern = re.compile(r'([A-Z ,\-]+) Precision: ([\d.]+)\s+\1 Recall: ([\d.]+)\s+\1 F1 Score: ([\d.]+)\s+True Num: (\d+)\s+Predicted Num: (\d+)\s+ENDCLASS', re.IGNORECASE)

    # Prepare to capture data
    data = []
    segments = language_pattern.split(content)

    # Iterate over each language section
    for i in range(1, len(segments), 2):
        language = segments[i].strip().lower()
        metrics_block = segments[i + 1]

        # Extract global metrics
        global_match = global_metrics_pattern.search(metrics_block)
        global_precision = global_recall = global_f1_score = length = None
        if global_match:
            global_precision = float(global_match.group(1))
            global_recall = float(global_match.group(2))
            global_f1_score = float(global_match.group(3))
            length = int(global_match.group(4))

        # Extract class-specific metrics
        for match in class_pattern.finditer(metrics_block):
            category = match.group(1).strip()
            precision = float(match.group(2))
            recall = float(match.group(3))
            f1_score = float(match.group(4))
            true_num = int(match.group(5))
            predicted_num = int(match.group(6))

            # Add to data list with global metrics
            data.append({
                'Language': language,
                'Category': category,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score,
                'True Num': true_num,
                'Predicted Num': predicted_num,
                'Global Precision': global_precision,
                'Global Recall': global_recall,
                'Global F1 Score': global_f1_score
            })

    # Convert the collected data to a DataFrame
    return pd.DataFrame(data)

# Reading the content from the new file
with open("attacks/google_before.log", "r") as file:
    content = file.read()

# Parse the file content with the updated approach
parsed_data_with_markers = parse_file_with_markers(content)

# Show the first few rows of the parsed data to confirm correct extraction
print(parsed_data_with_markers.head())

# Save the parsed data to a CSV for further analysis if needed
parsed_data_with_markers.to_csv("parsed_classification_metrics.csv", index=False)
