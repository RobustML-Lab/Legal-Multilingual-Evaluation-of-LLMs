import requests
from bs4 import BeautifulSoup
import json

# The base URL for the EuroVoc categories
base_url = "https://eur-lex.europa.eu/browse/eurovoc.html?locale="

languages = ['en', 'de', 'fr', 'it', 'es', 'pl', 'ro', 'nl', 'el', 'hu', 'pt', 'cs',
             'sv', 'bg', 'da', 'fi', 'sk', 'lt', 'hr', 'sl', 'et', 'lv', 'mt']

# Dictionary to store categories by language
eurovoc_categories = {}

# Loop through each language
for lang in languages:
    # Request the page for the specified language
    url = base_url + lang
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the category list by identifying the appropriate HTML structure (use developer tools to inspect it)
    categories = []
    for item in soup.select("ul.browseTree li a.py"):  # Modify selector as per HTML structure
        category = item.get_text(strip=True)
        if category:
            categories.append(category)

    # Store the categories in the dictionary
    eurovoc_categories[lang] = categories

# Save the results to a.py JSON file
with open("output/eurovoc_categories.json", "w", encoding="utf-8") as file:
    json.dump(eurovoc_categories, file, ensure_ascii=False, indent=4)

# Function to load categories for a.py specific language
def get_label_options(lang_code):
    with open("output/eurovoc_categories.json", "r", encoding="utf-8") as file:
        categories = json.load(file).get(lang_code, [])
    # Convert options to lowercase
    return [option.lower() for option in categories]

# Example usage
label_options = get_label_options("en")
print(label_options)  # Now label_options contains the categories in lowercase
