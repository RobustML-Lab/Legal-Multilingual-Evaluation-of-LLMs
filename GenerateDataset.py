import requests
import time
import csv
import json
from langdetect import detect, detect_langs, LangDetectException
import re

API_TOKEN = 'ea05f0e5-3dba-41f9-ad13-b86c780e4fb1'
BASE_URL = 'https://dataex.ohchr.org/uhri/api'
HEADERS = {'x-uhri-api-public-token': API_TOKEN}

LANGUAGES = ['en', 'fr', 'es', 'ar', 'ru', 'zh']
LANGUAGES_DETECT = ['en', 'fr', 'es', 'ar', 'ru', 'zh-cn']
COUNTRY_ID = 'a84e36d6-b59d-4b04-a9e4-d9deae2d3142'

def search_recommendations(mechanismId, documentId, index):

    recommendations = []
    result = []

    url = f"{BASE_URL}/search/mechanism_result"
    payload = {
        "mechanismId": mechanismId,
        "currentRecoDisplayed": index,
        "countries": [],
        "mechanisms": [],
        "affectedPersons": [],
        "sdgs": [],
        "themes": [],
        "documentCodes": [documentId],
        "documentTypes": [],
        "fromDate": None,
        "toDate": None,
        "searchText": "",
        "uprCycles": [],
        "uprPositions": [],
        "uprStates": []
    }
    for lang in LANGUAGES:
        response = requests.post(url, headers=HEADERS, json=payload, params={"culture": lang}).json()["listOfRecommendations"]
        recommendations.append(response)

    i = 0

    while (i < len(recommendations[0])):
        annotation_ids = [rec[i]["annotationId"] for rec in recommendations]
        annot_set = set(annotation_ids)
        if(len(annot_set) == 1):
            toContinue = False
            for j in range(6):
                text = recommendations[j][i]["text"]
                clean_text = re.sub(r"<.*?>", "", text)
                recommendations[j][i]["text"] = clean_text
                # if (len(text) > 100):
                #     text = text[50:90]
                try:
                    lang = detect(clean_text)
                except LangDetectException:
                    continue

                print("detected ", lang, "real ", LANGUAGES_DETECT[j])
                if (lang == "en" and j > 0):
                    toContinue = True
                    break
            if(toContinue):
                i += 1
                continue

            recommendations[0][i]["en"] = recommendations[0][i].pop("text")
            recommendations[0][i]["fr"] = recommendations[1][i]["text"]
            recommendations[0][i]["es"] = recommendations[2][i]["text"]
            recommendations[0][i]["ar"] = recommendations[3][i]["text"]
            recommendations[0][i]["ru"] = recommendations[4][i]["text"]
            recommendations[0][i]["zh"] = recommendations[5][i]["text"]
            result.append(recommendations[0][i])
        else:
            print("NOT 1 ann")
        i += 1

    return result

def get_recommendations(mechanism_id):
    url = f"{BASE_URL}/search/mechanism_result"
    payload = {
        "mechanismId": mechanism_id,
        "currentRecoDisplayed": 0,
        "countries": [COUNTRY_ID],
        "mechanisms": [],
        "affectedPersons": [],
        "sdgs": [],
        "themes": [],
        "documentCodes": [],
        "documentTypes": [],
        "fromDate": None,
        "toDate": None,
        "searchText": "",
        "uprCycles": [],
        "uprPositions": [],
        "uprStates": []
    }
    response = requests.post(url, headers=HEADERS, json=payload, params={"culture": "en"})
    return response.json().get("listOfRecommendations", [])

def fetch_multilingual_texts(annotation_id):
    results = {}
    for lang in LANGUAGES:
        url = f"{BASE_URL}/search/mechanism_result"
        payload = {
            "mechanismId": "",  # must match original mechanism
            "currentRecoDisplayed": 0,
            "countries": [COUNTRY_ID],
            "mechanisms": [],
            "affectedPersons": [],
            "sdgs": [],
            "themes": [],
            "documentCodes": [],
            "documentTypes": [],
            "fromDate": None,
            "toDate": None,
            "searchText": "",
            "uprCycles": [],
            "uprPositions": [],
            "uprStates": []
        }
        # This is a workaround: need to scan for the specific recommendation again in each language
        recs = get_recommendations_by_lang(lang)
        for rec in recs:
            if rec["annotationId"] == annotation_id:
                results[lang] = rec["text"]
                break
        time.sleep(0.5)
    return results if len(results) == len(LANGUAGES) else None

def get_recommendations_by_lang(lang):
    # Re-run main search by language
    url = f"{BASE_URL}/search/mechanism_result"
    payload = {
        "mechanismId": "",  # will set dynamically
        "currentRecoDisplayed": 0,
        "countries": [COUNTRY_ID],
        "mechanisms": [],
        "affectedPersons": [],
        "sdgs": [],
        "themes": [],
        "documentCodes": [],
        "documentTypes": [],
        "fromDate": None,
        "toDate": None,
        "searchText": "",
        "uprCycles": [],
        "uprPositions": [],
        "uprStates": []
    }
    response = requests.post(url, headers=HEADERS, json=payload, params={"culture": lang})
    return response.json().get("listOfRecommendations", [])

print("Fetching mechanisms...")
# mechanisms = search_recommendations()
# final_data = []
#
# for mech in mechanisms:
#     mech_id = mech["body"]
#     print(f"Processing mechanism: {mech_id}")
#     recs = get_recommendations(mech_id)
#
#     for rec in recs:
#         annotation_id = rec["annotationId"]
#         multilingual_texts = fetch_multilingual_texts(annotation_id)
#         if multilingual_texts:
#             final_data.append({
#                 "id": annotation_id,
#                 **multilingual_texts
#             })
#
# with open("multilingual_recommendations.csv", "w", newline='', encoding='utf-8') as f:
#     writer = csv.DictWriter(f, fieldnames=["id"] + LANGUAGES)
#     writer.writeheader()
#     for row in final_data:
#         writer.writerow(row)

URL_FILTER_LIST = f"{BASE_URL}/FilterList"
response = requests.get(URL_FILTER_LIST, headers=HEADERS).json()

countries = response["countries"]

print(countries)

# country_ids = [country["id"] for country in countries]
#
# go = False
#
# for country in country_ids:
#
#     if(country == "9309b34d-6fc7-4867-be1c-210dfa42c0d1"):
#         go = True
#
#     if(not go):
#         continue
#
#     URL_COUNTRY_INFO = f"{BASE_URL}/Country/country_info/{country}"
#     response = requests.get(URL_COUNTRY_INFO, headers=HEADERS).json()
#
#     country_documents = response["countryDocument"]
#
#     for document in country_documents:
#
#         print(document["documentSymbol"])
#
#         documentId = document["documentId"]
#
#         URL_DOCUMENT_INFO = f"{BASE_URL}/Document/document_info/{documentId}"
#         response = requests.get(URL_DOCUMENT_INFO, headers=HEADERS).json()
#         languages = [lang["language"] for lang in response["languages"]]
#         lang_set = set(languages)
#         print(languages)
#         print(lang_set)
#
#         if(len(lang_set)) < 6:
#             continue
#
#         index = 0
#         total = document['numberOfRecommendations'] + document["numberOfConcerns"]
#         print(total)
#
#         while index < total:
#             recommendation_list = search_recommendations(document["mechanismId"], documentId, index)
#             with open("recommendations5.jsonl", "a", encoding="utf-8") as f:
#                 for recommendation in recommendation_list:
#                     json.dump(recommendation, f, ensure_ascii=False)
#                     f.write("\n")
#             index += 40
#
#
# print("Done! Saved to multilingual_recommendations.csv")
#
