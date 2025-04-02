from collections import Counter

import requests
import time
import csv
import json
from langdetect import detect, detect_langs, LangDetectException
import re
import pandas as pd

API_TOKEN = 'ea05f0e5-3dba-41f9-ad13-b86c780e4fb1'
BASE_URL = 'https://dataex.ohchr.org/uhri/api'
HEADERS = {'x-uhri-api-public-token': API_TOKEN}

LANGUAGES = ['en', 'fr', 'es', 'ar', 'ru', 'zh']
LANGUAGES_DETECT = ['en', 'fr', 'es', 'ar', 'ru', 'zh-cn']

# URL_FILTER_LIST = f"{BASE_URL}/FilterList"
# response = requests.get(URL_FILTER_LIST, headers=HEADERS).json()
#
# with open("SDGs.jsonl", "a", encoding="utf-8") as f:
#     for sdg in response["sdGs"]:
#         json.dump(sdg, f, ensure_ascii=False)
#         f.write("\n")

data = []

files = [
    "filtered_recommendations.jsonl",
    "filtered_recommendations2.jsonl",
    "filtered_recommendations3.jsonl",
    "filtered_recommendations4.jsonl",
    "recommendations5.jsonl"
]

# Read all JSONL files line by line
for file_path in files:
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError:
                print(f"Skipping invalid line in {file_path}")

sdg_map = {}

with open("SDGs_main.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            # Assuming format: {"id": "...", "text": "1 - NO POVERTY"}
            sdg_number = int(obj["text"].split(" - ")[0])
            sdg_map[obj["id"]] = sdg_number
        except Exception as e:
            print(f"Skipping line: {line.strip()} — Error: {e}")

# Convert to DataFrame
df = pd.DataFrame(data)

def clean_sdgs_ids(sdg_list):
    if isinstance(sdg_list, list):
        return [sdg for sdg in sdg_list if sdg in sdg_map]
    return []

# Apply the cleaning function
df["sdgsIds"] = df["sdgsIds"].apply(clean_sdgs_ids)

filtered_df = df[
    df["sdgsIds"].apply(lambda x: isinstance(x, list) and len(x) == 1)
    & df["en"].apply(lambda text: isinstance(text, str) and len(text.strip()) > 1000)
]
all_sdgs = [sdg_id[0] for sdg_id in filtered_df["sdgsIds"] if isinstance(sdg_id, list) and len(sdg_id) == 1]
sdg_counts = Counter(all_sdgs)

print(sdg_counts)

# Sample 15 entries for each valid SDG
final_samples = []
for sdg_id in sdg_counts.keys():
    sdg_entries = filtered_df[filtered_df["sdgsIds"].apply(lambda x: x[0] == sdg_id)]
    sample_size = min(len(sdg_entries), 25)
    sampled = sdg_entries.sample(n=sample_size, random_state=42)
    final_samples.append(sampled)

# Combine all the sampled entries
result_df = pd.concat(final_samples, ignore_index=True)

# Save the result
result_df.to_json("sampled_sdgs_one_label_large_text.jsonl",
                  orient="records",
                  lines=True,
                  force_ascii=False)