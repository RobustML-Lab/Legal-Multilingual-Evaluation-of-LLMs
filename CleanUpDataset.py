import json

input_file = "recommendations4.jsonl"
output_file = "filtered_recommendations4.jsonl"
excluded_country_id = "3251dccf-0213-458d-bf38-016991aa8b57"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            obj = json.loads(line)
            if obj.get("countryIds") != [excluded_country_id]:
                json.dump(obj, outfile, ensure_ascii=False)
                outfile.write("\n")
        except json.JSONDecodeError:
            print("Skipping invalid JSON line.")
