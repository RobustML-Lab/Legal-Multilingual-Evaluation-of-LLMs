from transformers import pipeline

# Load zero-shot pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# SDG labels
sdg_labels = [
    "NO POVERTY",
    "ZERO HUNGER",
    "GOOD HEALTH AND WELL-BEING",
    "QUALITY EDUCATION",
    "GENDER EQUALITY",
    "CLEAN WATER AND SANITATION",
    "AFFORDABLE AND CLEAN ENERGY",
    "DECENT WORK AND ECONOMIC GROWTH",
    "INDUSTRY, INNOVATION AND INFRASTRUCTURE",
    "REDUCED INEQUALITIES",
    "SUSTAINABLE CITIES AND COMMUNITIES",
    "RESPONSIBLE CONSUMPTION AND PRODUCTION",
    "CLIMATE ACTION",
    "LIFE BELOW WATER",
    "LIFE ON LAND",
    "PEACE, JUSTICE AND STRONG INSTITUTIONS",
    "PARTNERSHIPS FOR THE GOALS"
]

# Example text
text = """
11.The Committee notes that under the eighteenth amendment to the Constitution (2010), the provinces have been granted greater autonomy, as the federal Government has transferred to them the policymaking authority on crucial sectors such as health, education, and employment as well as on all matters related to the advancement of women. However, the Committee is concerned about the governance challenges embodied in the devolution of powers, including the integration and coordination of policies aimed at the advancement of women, from the national to the provincial level. It is also concerned that the State party lacks the capacity to put in place an efficient mechanism to ensure that the provincial governments establish legal and other measures to fully implement the Convention in a coherent and consistent manner. It is further concerned that the different levels of authority and competence within the State party due to the devolution of powers may result in a differentiated application of the law."""

# Run zero-shot prediction
result = classifier(text, sdg_labels)

# Print top result
print("Predicted SDG:", result["labels"][0])
