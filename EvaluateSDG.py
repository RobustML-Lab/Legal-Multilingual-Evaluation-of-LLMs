from collections import Counter

import pandas as pd
import json
import ast
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

# # === Load files ===
# df = pd.read_csv("classification_results.csv")
#
# # Build SDG GUID-to-number mapping
# sdg_map = {}
# with open("SDGs_main.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         try:
#             obj = json.loads(line)
#             sdg_number = int(obj["text"].split(" - ")[0])
#             sdg_map[obj["id"]] = sdg_number
#         except Exception as e:
#             print(f"Skipping line: {line.strip()} — Error: {e}")
#
# # === Helper functions ===
#
# def map_sdg_ids_to_numbers(sdg_ids_str):
#     try:
#         sdg_ids = ast.literal_eval(sdg_ids_str)
#     except Exception:
#         return []
#     return [sdg_map[sdg_id] for sdg_id in sdg_ids if sdg_id in sdg_map]
#
# def extract_sdg_number(pred):
#     if pd.isna(pred):
#         return []
#
#     matches = re.findall(r"\b(1[0-7]|[1-9])\b", str(pred))  # match numbers 1–17
#     counts = Counter(int(m) for m in matches)
#
#     if not counts:
#         return []
#
#     most_common = counts.most_common()
#     top_count = most_common[0][1]
#
#     # Check if there's a unique top frequency
#     top_numbers = [num for num, count in most_common if count == top_count]
#
#     if len(top_numbers) == 1:
#         return [top_numbers[0]]
#     else:
#         return []
#
# # === Clean & process data ===
# df["true_sdg_nums"] = df["sdg_categories"].apply(map_sdg_ids_to_numbers)
# df["predicted_sdg_nums"] = df["predicted"].apply(extract_sdg_number)
#
# df = df[df["true_sdg_nums"].map(len) > 0]
# df = df[df["predicted_sdg_nums"].map(len) > 0]
#
# # Binarize SDG labels
# all_sdg_numbers = sorted(set(sdg_map.values()))
# mlb = MultiLabelBinarizer(classes=all_sdg_numbers)
# mlb.fit([all_sdg_numbers])
#
# # === Evaluate per language ===
# results = []
#
# for lang, group in df.groupby("language"):
#     y_true = mlb.transform(group["true_sdg_nums"])
#     y_pred = mlb.transform(group["predicted_sdg_nums"])
#
#     acc = (y_true == y_pred).all(axis=1).mean()
#     prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
#     rec = recall_score(y_true, y_pred, average="micro", zero_division=0)
#     f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
#
#     results.append({
#         "language": lang,
#         "accuracy": round(acc, 4),
#         "precision": round(prec, 4),
#         "rec": round(rec, 4),
#         "f1": round(f1, 4)
#     })
#
# results_df = pd.DataFrame(results)
# print(results_df)
#
# # === Debug export ===
# debug_df = df[["annotationId", "language", "true_sdg_nums", "predicted_sdg_nums", "predicted"]]
# debug_df.to_csv("sdg_predictions_debug.csv", index=False)
#
# # === Plot ===
# results_df.set_index("language", inplace=True)
# results_df.plot(kind="bar", figsize=(10, 6))
# plt.title("SDG Classification Metrics per Language")
# plt.ylabel("Score")
# plt.ylim(0, 1.05)
# plt.xticks(rotation=45)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig("sdg_language_metrics.png", dpi=300)
# plt.show()
#
#
# ######################################################################
#
# # === Load files ===
# df = pd.read_csv("llama_character_en.csv")
#
# # Build SDG GUID-to-number mapping
# sdg_map = {}
# with open("SDGs_main.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         try:
#             obj = json.loads(line)
#             sdg_number = int(obj["text"].split(" - ")[0])
#             sdg_map[obj["id"]] = sdg_number
#         except Exception as e:
#             print(f"Skipping line: {line.strip()} — Error: {e}")
#
# # === Helper functions ===
#
# def map_sdg_ids_to_numbers(sdg_ids_str):
#     try:
#         sdg_ids = ast.literal_eval(sdg_ids_str)
#     except Exception:
#         return []
#     return [sdg_map[sdg_id] for sdg_id in sdg_ids if sdg_id in sdg_map]
#
# def extract_sdg_number(pred):
#     if pd.isna(pred):
#         return []
#
#     matches = re.findall(r"\b(1[0-7]|[1-9])\b", str(pred))  # match numbers 1–17
#     counts = Counter(int(m) for m in matches)
#
#     if not counts:
#         return []
#
#     most_common = counts.most_common()
#     top_count = most_common[0][1]
#
#     # Check if there's a unique top frequency
#     top_numbers = [num for num, count in most_common if count == top_count]
#
#     if len(top_numbers) == 1:
#         return [top_numbers[0]]
#     else:
#         return []
#
# # === Clean & process data ===
# df["true_sdg_nums"] = df["sdg_categories"].apply(map_sdg_ids_to_numbers)
# df["predicted_sdg_nums"] = df["predicted"].apply(extract_sdg_number)
#
# df = df[df["true_sdg_nums"].map(len) > 0]
# df = df[df["predicted_sdg_nums"].map(len) > 0]
#
# # Binarize SDG labels
# all_sdg_numbers = sorted(set(sdg_map.values()))
# mlb = MultiLabelBinarizer(classes=all_sdg_numbers)
# mlb.fit([all_sdg_numbers])
#
# # === Evaluate per language ===
# results = []
#
# for lang, group in df.groupby("language"):
#     y_true = mlb.transform(group["true_sdg_nums"])
#     y_pred = mlb.transform(group["predicted_sdg_nums"])
#
#     acc = (y_true == y_pred).all(axis=1).mean()
#     prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
#     rec = recall_score(y_true, y_pred, average="micro", zero_division=0)
#     f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
#
#     results.append({
#         "language": lang,
#         "accuracy": round(acc, 4),
#         # "precision": round(prec, 4),
#         # "rec": round(rec, 4),
#         # "f1": round(f1, 4)
#     })
#
# results_df = pd.DataFrame(results)
# print(results_df)
#
# # === Debug export ===
# debug_df = df[["annotationId", "language", "true_sdg_nums", "predicted_sdg_nums", "predicted"]]
# debug_df.to_csv("sdg_predictions_debug.csv", index=False)
#
# # === Plot ===
# results_df.set_index("language", inplace=True)
# results_df.plot(kind="bar", figsize=(10, 6))
# plt.title("SDG Classification Metrics per Language")
# plt.ylabel("Score")
# plt.ylim(0, 1.05)
# plt.xticks(rotation=45)
# plt.grid(axis="y", linestyle="--", alpha=0.7)
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.savefig("sdg_language_metrics.png", dpi=300)
# plt.show()

# === Load files ===
df = pd.read_csv("classification_results.csv")

# Build SDG GUID-to-number mapping
sdg_map = {}
with open("SDGs_main.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            sdg_number = int(obj["text"].split(" - ")[0])
            sdg_map[obj["id"]] = sdg_number
        except Exception as e:
            print(f"Skipping line: {line.strip()} — Error: {e}")

# === Helper functions ===

def map_sdg_ids_to_numbers(sdg_ids_str):
    try:
        sdg_ids = ast.literal_eval(sdg_ids_str)
    except Exception:
        return []
    return [sdg_map[sdg_id] for sdg_id in sdg_ids if sdg_id in sdg_map]

def extract_all_sdg_numbers(pred):
    if pd.isna(pred):
        return []
    matches = re.findall(r"\b(1[0-7]|[1-9])\b", str(pred))  # match numbers 1–17
    return sorted(set(int(m) for m in matches))  # unique, sorted list

# === Clean & process data ===
df["true_sdg_nums"] = df["sdg_categories"].apply(map_sdg_ids_to_numbers)
df["predicted_sdg_nums"] = df["predicted"].apply(extract_all_sdg_numbers)

# Remove rows with no true or predicted SDGs
df = df[df["true_sdg_nums"].map(len) > 0]
df = df[df["predicted_sdg_nums"].map(len) > 0]

# Binarize SDG labels
all_sdg_numbers = sorted(set(sdg_map.values()))
mlb = MultiLabelBinarizer(classes=all_sdg_numbers)
mlb.fit([all_sdg_numbers])

# === Evaluate per language ===
results = []

for lang, group in df.groupby("language"):
    y_true = mlb.transform(group["true_sdg_nums"])
    y_pred = mlb.transform(group["predicted_sdg_nums"])

    acc = (y_true == y_pred).all(axis=1).mean()
    prec = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    results.append({
        "language": lang,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "rec": round(rec, 4),
        "f1": round(f1, 4)
    })

results_df = pd.DataFrame(results)
print(results_df)

# === Debug export ===
debug_df = df[["annotationId", "language", "true_sdg_nums", "predicted_sdg_nums", "predicted"]]
debug_df.to_csv("sdg_predictions_debug.csv", index=False)

# === Plot ===
results_df.set_index("language", inplace=True)
results_df.plot(kind="bar", figsize=(10, 6))
plt.title("SDG Classification Metrics per Language")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("sdg_language_metrics.png", dpi=300)
plt.show()