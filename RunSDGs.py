import json
import time
from datetime import datetime

import google.generativeai as genai
import pandas as pd
import re

subjects_en = [
"1 - NO POVERTY",
"2 - ZERO HUNGER",
"3 - GOOD HEALTH AND WELL-BEING",
"4 - QUALITY EDUCATION",
"5 - GENDER EQUALITY",
"6 - CLEAN WATER AND SANITATION",
"7 - AFFORDABLE AND CLEAN ENERGY",
"8 - DECENT WORK AND ECONOMIC GROWTH",
"9 - INDUSTRY, INNOVATION AND INFRASTRUCTURE",
"10 - REDUCED INEQUALITIES",
"11 - SUSTAINABLE CITIES AND COMMUNITIES",
"12 - RESPONSIBLE CONSUMPTION AND PRODUCTION",
"13 - CLIMATE ACTION",
"14 - LIFE BELOW WATER",
"15 - LIFE ON LAND",
"16 - PEACE, JUSTICE AND STRONG INSTITUTIONS",
"17 - PARTNERSHIPS FOR THE GOALS"

]

subjects_ar = [
    "1 - القضاء على الفقر",
    "2 - القضاء على الجوع",
    "3 - الصحة الجيدة والرفاهية",
    "4 - التعليم الجيد",
    "5 - المساواة بين الجنسين",
    "6 - نظافة المياه والصرف الصحي",
    "7 - طاقة نظيفة وبتكلفة ميسورة",
    "8 - عمل لائق ونمو اقتصادي",
    "9 - صناعة، ابتكار وبنية التحتية",
    "10 - الحد من انعدام المساواة",
    "11 - مدن ومجتمعات مستدامة",
    "12 - استهلاك وإنتاج مسؤول",
    "13 - العمل المناخي",
    "14 - الحياة تحت الماء",
    "15 - الحياة البرية",
    "16 - سلام، عدالة ومؤسسات قوية",
    "17 - شراكات من أجل الأهداف"
]

subjects_ru = [
    "1 - ЛИКВИДАЦИЯ НИЩЕТЫ",
    "2 - ЛИКВИДАЦИЯ ГОЛОДА",
    "3 - ХОРОШЕЕ ЗДОРОВЬЕ И БЛАГОПОЛУЧИЕ",
    "4 - КАЧЕСТВЕННОЕ ОБРАЗОВАНИЕ",
    "5 - ГЕНДЕРНОЕ РАВЕНСТВО",
    "6 - ЧИСТАЯ ВОДА И САНИТАРИЯ",
    "7 - НЕДОРОГОСТОЯЩАЯ И ЧИСТАЯ ЭНЕРГИЯ",
    "8 - ДОСТОЙНАЯ РАБОТА И ЭКОНОМИЧЕСКИЙ РОСТ",
    "9 - ИНДУСТРИАЛИЗАЦИЯ, ИННОВАЦИЯ И ИНФРАСТРУКТУРА",
    "10 - УМЕНЬШЕНИЕ НЕРАВЕНСТВА",
    "11 - УСТОЙЧИВЫЕ ГОРОДА И НАСЕЛЕННЫЕ ПУНКТЫ",
    "12 - ОТВЕТСТВЕННОЕ ПОТРЕБЛЕНИЕ И ПРОИЗВОДСТВО",
    "13 - БОРЬБА С ИЗМЕНЕНИЕМ КЛИМАТА",
    "14 - СОХРАНЕНИЕ МОРСКИХ ЭКОСИСТЕМ",
    "15 - СОХРАНЕНИЕ ЭКОСИСТЕМ СУШИ",
    "16 - МИР, ПРАВОСУДИЕ И ЭФФЕКТИВНЫЕ ИНСТИТУТЫ",
    "17 - ПАРТНЕРСТВО В ИНТЕРЕСАХ УСТОЙЧИВОГО РАЗВИТИЯ"
]

subjects_es = [
    "1 - Pobreza",
    "2 - Cero Hambre",
    "3 - Buena Salud y Bienestar",
    "4 - Educación de Calidad",
    "5 - Igualdad de Género",
    "6 - Agua potable y Saneamiento",
    "7 - Energía Limpia y Accesible",
    "8 - Trabajo Decente y Crecimiento Económico",
    "9 - Industria, Innovación e Infraestructura",
    "10 - Desigualdades Reducidas",
    "11 - Ciudades y Comunidades Sostenibles",
    "12 - Producción y Consumo Responsables",
    "13 - Cambio Climático",
    "14 - Océanos",
    "15 - Bosques, desertificación y diversidad biológica",
    "16 - Paz, Justicia e Instituciones",
    "17 - Alianzas para los Objetivos"
]

subjects_fr = [
    "1 - PAS DE PAUVRETE",
    "2 - FAIM \"ZERO\"",
    "3 - BONNE SANTE ET BIEN-ETRE",
    "4 - EDUCATION DE QUALITE",
    "5 - EGALITE ENTRE LES SEXES",
    "6 - EAU PROPRE ET ASSAINISSEMENT",
    "7 - ENERGIE PROPRE ET D'UN COUT ABORDABLE",
    "8 - TRAVAIL DECENT ET CROISSANCE ECONOMIQUE",
    "9 - INDUSTRIE, INNOVATION ET INFRASTRUCTURE",
    "10 - INEGALITES REDUITES",
    "11 - VILLES ET COMMUNAUTES DURABLES",
    "12 - CONSOMMATION ET PRODUCTION RESPONSABLES",
    "13 - LUTTE CONTRE LES CHANGEMENTS CLIMATIQUES",
    "14 - VIE AQUATIQUE",
    "15 - VIE TERRESTRE",
    "16 - PAIX, JUSTICE ET INSTITUTIONS EFFICACES",
    "17 - PARTENARIATS POUR LA REALISATION DES OBJECTIFS"
]

subjects_ch = [
    "1. 无贫穷",
    "2. 零饥饿",
    "3. 良好健康与福祉",
    "4. 优质教育",
    "5. 性别平等",
    "6. 清洁饮水和卫生设施",
    "7. 可负担的清洁能源",
    "8. 体面工作和经济增长",
    "9. 产业、创新和基础设施",
    "10. 减少不平等",
    "11. 可持续城市和社区",
    "12. 负责任消费和生产",
    "13. 气候行动",
    "14. 水下生物",
    "15. 陆地生物",
    "16. 和平、正义与强大机构",
    "17. 促进目标实现的伙伴关系"
]

subjects_dict = {
    "en": subjects_en,
    "ar": subjects_ar,
    "es": subjects_es,
    "fr": subjects_fr,
    "ru": subjects_ru,
    "zh": subjects_ch
}

language = "ar"

csv_filename = "single_label_character_attack_" + language + ".csv"

api_key_1 = "AIzaSyD-hV3U76j5yumpZGqGS0pjdCC8YKMOj1U"
api_key_2 = "AIzaSyAZSb17dO7NCkzulicsbG85B57HJi_AtRM"
api_key_3 = "AIzaSyDyIzfSkmXR788Ku6sf-dsGVVJ7welGyR8"

data = []

file_path = "sampled_sdgs_attack_character_level_02.jsonl"

with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        try:
            obj = json.loads(line)
            data.append(obj)
        except json.JSONDecodeError:
            print(f"Skipping invalid line in {file_path}")

df = pd.DataFrame(data)

genai.configure(api_key=api_key_2)
model = genai.GenerativeModel("gemini-2.0-flash")

REQUESTS_PER_MINUTE = 15
REQUEST_WINDOW = 60  # seconds
request_timestamps = []  # Store request times

def enforce_rate_limit():
    global request_timestamps
    now = time.time()

    # Remove timestamps older than 60 seconds
    request_timestamps = [t for t in request_timestamps if now - t < REQUEST_WINDOW]

    if len(request_timestamps) >= REQUESTS_PER_MINUTE:
        # Calculate time to wait
        wait_time = max(0, REQUEST_WINDOW - (now - request_timestamps[0]))
        print(f"Rate limit reached! Waiting {round(wait_time, 2)} seconds...")
        time.sleep(wait_time)  # Wait before making the next request


minute = datetime.now().minute + 1
print(f"Current Minute: {minute}")
cnt = 0

file_exists = False

go = False
for index, row in df.iterrows():

    # if(not go):
    #     if (row["annotationId"] == "c7bf5393-2ed2-4edf-91e3-65bdbd756e13"):
    #         go = True
    #     continue

    text = row[language + "_char_attack"]
    sdgs = row["sdgsIds"]
    subjects_list = subjects_dict.get(language)
    subjects_str = ", ".join(subjects_list)

    message = re.sub(r'\n+', ' ', text).strip()

    if language == "en":
        prompt = "Classify the following text into one sustainable development goal category from the list: **Possible Categories**: [" + subjects_str + "]. Return only the single most relevant category for this text. Do not include anything else in your answer." + message

    elif language == "ar":
        prompt = "صنّف النص التالي ضمن فئة واحدة فقط من فئات أهداف التنمية المستدامة الواردة في القائمة التالية: **الفئات الممكنة**: [" + subjects_str + "]. أعد فقط الفئة الأكثر صلة بالنص. لا تضف أي معلومات أخرى في إجابتك." + message

    elif language == "es":
        prompt = "Clasifica el siguiente texto en una sola categoría de los Objetivos de Desarrollo Sostenible de la lista siguiente: **Categorías posibles**: [" + subjects_str + "]. Devuelve únicamente la categoría más relevante para este texto. No incluyas ninguna otra información en tu respuesta." + message

    elif language == "fr":
        prompt = "Classez le texte suivant dans une seule catégorie des objectifs de développement durable figurant dans la liste suivante : **Catégories possibles**: [" + subjects_str + "]. Retournez uniquement la catégorie la plus pertinente pour ce texte. N'ajoutez aucune autre information dans votre réponse." + message

    elif language == "ru":
        prompt = "Классифицируйте следующий текст в одну категорию из целей устойчивого развития из следующего списка: **Возможные категории**: [" + subjects_str + "]. Верните только одну наиболее подходящую категорию. Не добавляйте никакой другой информации в ответ." + message

    elif language == "zh":
        prompt = "将以下文本分类为以下列表中一个可持续发展目标类别：**可能的类别**: [" + subjects_str + "]. 仅返回最相关的一个类别。请勿包含任何其他信息。" + message

    enforce_rate_limit()

    print(prompt)

    try:
        response = model.generate_content(prompt)
        predicted = response.text.strip()
    except Exception as e:
        print(f"Error processing entry {row['annotationId']}: {e}")
        predicted = "Error"

    request_timestamps.append(time.time())

    result = {
        "annotationId": row["annotationId"],
        "language": language,
        "sdg_categories": sdgs,
        "predicted": predicted,
    }

    df_result = pd.DataFrame([result])
    df_result.to_csv(csv_filename, mode='a', header=not file_exists, index=False)

    file_exists = True  # Ensure file is marked as existing